"""
BERT in reinem PyTorch (ohne nn.MultiheadAttention / nn.TransformerEncoder).

Ziel: Self-Attention und den BERT-Encoder verstehen und per Masked Language
Modeling (MLM) trainieren.

Nutzung:
    python bert.py --smoke                         # Schnelltest, synthetische Daten
    python bert.py --text corpus.txt --pack        # eigener Text, ein Absatz pro Zeile
    python bert.py --smoke --plot attn.png         # zusätzlich Attention-Heatmaps speichern

Vereinfachungen gegenüber Devlin et al. (2018):
  - Regex-Wort-Tokenizer statt WordPiece (Schnittstelle bleibt gleich)
  - nur MLM, kein Next Sentence Prediction (vgl. RoBERTa)
  - kleines Modell, damit Training auf CPU/Consumer-GPU möglich ist
"""

import argparse
import math
import random
import re
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.utils.device import get_device
from torch.utils.data import DataLoader, Dataset

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIALS = [PAD, UNK, CLS, SEP, MASK]
_TOKEN_RE = re.compile(r"\[mask\]|\w+|[^\w\s]")


# ----------------------------------------------------------------------------
# 1. Tokenizer
# ----------------------------------------------------------------------------
class Tokenizer:
    """Wort-Tokenizer; Satzzeichen sind eigene Tokens. Für echtes BERT-Verhalten
    durch WordPiece ersetzen, solange encode()/itos/stoi gleich bleiben."""

    def __init__(self, lines, max_vocab=8000):
        counts = Counter(t for line in lines for t in self.split(line))
        words = [w for w, _ in counts.most_common(max_vocab - len(SPECIALS))]
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id = self.stoi[PAD], self.stoi[UNK]
        self.cls_id, self.sep_id, self.mask_id = self.stoi[CLS], self.stoi[SEP], self.stoi[MASK]

    @staticmethod
    def split(line):
        # "[MASK]" bleibt als ein Token erhalten (kleingeschrieben durch lower())
        return _TOKEN_RE.findall(line.lower())

    def __len__(self):
        return len(self.itos)

    def encode(self, line):
        out = []
        for t in self.split(line):
            out.append(self.mask_id if t == "[mask]" else self.stoi.get(t, self.unk_id))
        return out


# ----------------------------------------------------------------------------
# 2. Modell
# ----------------------------------------------------------------------------
@dataclass
class BertConfig:
    vocab_size: int
    max_len: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    pad_id: int = 0


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention von Hand.

    B=Batch, T=Sequenzlänge, H=Köpfe, D=d_model, d=D/H.
    Ohne return_attn läuft der schnelle Pfad (F.scaled_dot_product_attention,
    mathematisch identisch); mit return_attn wird die Rechnung explizit
    ausgeschrieben, um die Gewichte inspizieren zu können.
    """

    def __init__(self, cfg: BertConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.d = cfg.d_model // cfg.n_heads
        self.p = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # Q, K, V in einer Matrix
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask, return_attn=False):
        # x: (B, T, D); key_padding_mask: (B, T), True = echtes Token
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.d).transpose(1, 2) for t in (q, k, v))  # (B,H,T,d)
        attn = None
        if return_attn:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)  # (B,H,T,T)
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
            attn = scores.softmax(dim=-1)
            out = self.drop(attn) @ v
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_padding_mask[:, None, None, :],
                dropout_p=self.p if self.training else 0.0,
            )
        out = self.out(out.transpose(1, 2).reshape(B, T, D))
        return (out, attn) if return_attn else out


class EncoderLayer(nn.Module):
    """Post-LN wie im Original-BERT: x = LN(x + Sublayer(x))."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.attn = SelfAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=1e-12)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=1e-12)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask, return_attn=False):
        if return_attn:
            a, attn = self.attn(x, mask, return_attn=True)
        else:
            a, attn = self.attn(x, mask), None
        x = self.ln1(x + self.drop(a))
        x = self.ln2(x + self.drop(self.ff(x)))
        return (x, attn) if return_attn else x


class Bert(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)  # gelernt
        self.seg_emb = nn.Embedding(2, cfg.d_model)  # Segment A/B (hier ungenutzt)
        self.emb_ln = nn.LayerNorm(cfg.d_model, eps=1e-12)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))

        # MLM-Kopf; Ausgabegewichte mit tok_emb geteilt (weight tying)
        self.mlm_dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlm_ln = nn.LayerNorm(cfg.d_model, eps=1e-12)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        self.apply(self._init)
        with torch.no_grad():  # Init überschreibt die Pad-Zeile
            self.tok_emb.weight[cfg.pad_id].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, ids, segments=None, return_attn=False):
        B, T = ids.shape
        assert self.cfg.max_len >= T, f"Sequenz ({T}) länger als max_len ({self.cfg.max_len})"
        mask = ids != self.cfg.pad_id
        pos = torch.arange(T, device=ids.device)[None, :]
        if segments is None:
            segments = torch.zeros_like(ids)
        x = self.emb_drop(self.emb_ln(self.tok_emb(ids) + self.pos_emb(pos) + self.seg_emb(segments)))
        attns = []
        for layer in self.layers:
            if return_attn:
                x, a = layer(x, mask, return_attn=True)
                attns.append(a)
            else:
                x = layer(x, mask)
        return (x, attns) if return_attn else x

    def forward(self, ids, segments=None):
        h = self.mlm_ln(F.gelu(self.mlm_dense(self.encode(ids, segments))))
        return h @ self.tok_emb.weight.T + self.mlm_bias  # (B, T, V)


# ----------------------------------------------------------------------------
# 3. Daten + MLM-Masking
# ----------------------------------------------------------------------------
def mlm_mask(ids, tok: Tokenizer, p=0.15, generator=None):
    """BERT-Masking: ~15 % der Tokens; davon 80 % [MASK], 10 % zufällig, 10 % unverändert.
    Jede Sequenz bekommt mindestens ein Ziel, sonst kann der Loss NaN werden
    (cross_entropy über null Ziele). Labels sind -100 an nicht gewählten Stellen."""
    ids = ids.clone()
    labels = ids.clone()
    special = (ids == tok.pad_id) | (ids == tok.cls_id) | (ids == tok.sep_id)

    noise = torch.rand(ids.shape, generator=generator)
    noise[special] = 2.0  # nie wählen
    chosen = noise < p
    # Garantie: pro Sequenz mindestens die Position mit dem kleinsten Rauschwert
    first = noise.argmin(dim=1)
    chosen[torch.arange(ids.size(0)), first] = noise[torch.arange(ids.size(0)), first] < 2.0
    labels[~chosen] = -100

    r = torch.rand(ids.shape, generator=generator)
    ids[chosen & (r < 0.8)] = tok.mask_id
    rand_pos = chosen & (r >= 0.8) & (r < 0.9)
    ids[rand_pos] = torch.randint(len(SPECIALS), len(tok), (int(rand_pos.sum()),), generator=generator)
    return ids, labels


class TextDataset(Dataset):
    """pack=False: eine Zeile = eine Sequenz (gepaddet).
    pack=True: alle Tokens aneinanderhängen und in volle Blöcke teilen (wie RoBERTa);
    spart Padding bei kurzen Zeilen."""

    def __init__(self, lines, tok: Tokenizer, max_len, pack=False):
        body_len = max_len - 2
        self.items = []
        if pack:
            stream = []
            for line in lines:
                stream += tok.encode(line)
            chunks = [stream[i : i + body_len] for i in range(0, len(stream) - body_len + 1, body_len)]
        else:
            chunks = [tok.encode(line)[:body_len] for line in lines]
        for body in chunks:
            if len(body) < 3:
                continue
            seq = [tok.cls_id] + body + [tok.sep_id]
            seq += [tok.pad_id] * (max_len - len(seq))
            self.items.append(torch.tensor(seq))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def synthetic_corpus(n=3000, seed=0):
    """Das Verb hängt vom Subjekt ab, das Objekt vom Verb. Nur wer den Kontext
    (per Attention) nutzt, kann ein maskiertes Wort sicher vorhersagen."""
    rng = random.Random(seed)
    subj_verb = {
        "the cat": "chases",
        "the dog": "guards",
        "a bird": "builds",
        "my friend": "reads",
        "the teacher": "grades",
    }
    verb_obj = {
        "chases": "the ball",
        "guards": "the house",
        "builds": "a nest",
        "reads": "a book",
        "grades": "the exam",
    }
    out = []
    for _ in range(n):
        s = rng.choice(list(subj_verb))
        v = subj_verb[s]
        out.append(f"{s} {v} {verb_obj[v]} .")
    return out


# ----------------------------------------------------------------------------
# 4. Training / Evaluation
# ----------------------------------------------------------------------------
def make_optimizer(model, lr, weight_decay=0.01):
    """Kein Weight Decay auf Biases und LayerNorm (Standard bei BERT)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-6,
    )


@torch.no_grad()
def evaluate(model, loader, tok, device, seed=1234):
    """Validierung mit festem Masking, damit Werte zwischen Epochen vergleichbar sind."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    total_loss = correct = n = 0
    for batch in loader:
        inp, labels = mlm_mask(batch, tok, generator=g)
        inp, labels = inp.to(device), labels.to(device)
        logits = model(inp)
        tgt = labels != -100
        total_loss += F.cross_entropy(logits[tgt], labels[tgt], reduction="sum").item()
        correct += (logits[tgt].argmax(-1) == labels[tgt]).sum().item()
        n += tgt.sum().item()
    return total_loss / n, correct / n


def train(model, train_loader, val_loader, tok, epochs, lr, device, warmup_frac=0.1, log_every=50):
    opt = make_optimizer(model, lr)
    total = epochs * len(train_loader)
    warmup = min(max(1, int(warmup_frac * total)), max(1, total - 1))
    # linearer Warmup, danach linearer Abfall auf 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup))
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    step, run = 0, []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inp, labels = mlm_mask(batch, tok)
            inp, labels = inp.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(inp)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run.append(loss.item())
            if step % log_every == 0:
                print(f"step {step}/{total}  train-loss {sum(run) / len(run):.3f}  lr {sched.get_last_lr()[0]:.2e}")
                run = []
        if val_loader is not None:
            vl, acc = evaluate(model, val_loader, tok, device)
            print(f"== epoch {epoch + 1}/{epochs}  val-loss {vl:.3f}  ppl {math.exp(vl):.1f}  mlm-acc {acc:.1%}")


# ----------------------------------------------------------------------------
# 5. Inspektion
# ----------------------------------------------------------------------------
@torch.no_grad()
def fill_mask(model, tok, sentence, device, topk=5):
    """Satz mit [MASK] eingeben, Top-k-Vorhersagen je Maske anzeigen (Spezialtokens ausgeblendet)."""
    model.eval()
    ids = [tok.cls_id] + tok.encode(sentence) + [tok.sep_id]
    probs = model(torch.tensor([ids], device=device))[0].softmax(-1)
    probs[:, : len(SPECIALS)] = 0
    for i, t in enumerate(ids):
        if t == tok.mask_id:
            top = probs[i].topk(topk)
            print(f"  Position {i}:", [(tok.itos[j], round(p.item(), 3)) for p, j in zip(top.values, top.indices)])


@torch.no_grad()
def plot_attention(model, tok, sentence, path, device, layer=-1):
    """Attention-Heatmaps aller Köpfe einer Schicht für einen Satz."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    ids = [tok.cls_id] + tok.encode(sentence) + [tok.sep_id]
    _, attns = model.encode(torch.tensor([ids], device=device), return_attn=True)
    a = attns[layer][0].cpu()  # (H, T, T)
    labels = [tok.itos[i] for i in ids]
    fig, axes = plt.subplots(1, a.size(0), figsize=(3.2 * a.size(0), 3.6))
    for h, ax in enumerate(axes if a.size(0) > 1 else [axes]):
        ax.imshow(a[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"Kopf {h}", fontsize=9)
    fig.suptitle(f"Attention, Schicht {layer % len(model.layers)} (Zeile = Query, Spalte = Key)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("Attention-Plot gespeichert:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Textdatei, ein Satz/Absatz pro Zeile")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pack", action="store_true", help="Tokens zu vollen Blöcken packen")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None, help="Checkpoint-Pfad")
    ap.add_argument("--plot", type=str, default=None, help="Pfad für Attention-Heatmap (PNG)")
    ap.add_argument("--probe", type=str, default=None, help="Satz mit [MASK] zum Testen")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    if args.smoke or not args.text:
        lines = synthetic_corpus()
        probe = args.probe or "the cat [MASK] the ball ."
    else:
        with open(args.text, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        probe = args.probe
    random.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_frac))
    val_lines, train_lines = lines[:n_val], lines[n_val:]

    tok = Tokenizer(train_lines)  # Vokabular nur aus Trainingsdaten
    cfg = BertConfig(vocab_size=len(tok), max_len=args.max_len, pad_id=tok.pad_id)
    if args.smoke:
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff = 64, 4, 2, 256

    model = Bert(cfg).to(device)
    print(
        f"Parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M | Vokabular: {len(tok)} | Gerät: {device}"
    )
    tr = DataLoader(
        TextDataset(train_lines, tok, cfg.max_len, args.pack), batch_size=args.batch, shuffle=True, drop_last=False
    )
    va = DataLoader(TextDataset(val_lines, tok, cfg.max_len, args.pack), batch_size=args.batch)
    train(model, tr, va, tok, args.epochs, args.lr, device)

    if probe:
        print("Fill-Mask:", probe)
        fill_mask(model, tok, probe, device)
    if args.plot:
        plot_attention(model, tok, probe.replace("[MASK]", "chases") if probe else lines[0], args.plot, device)
    if args.save:
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "vocab": tok.itos}, args.save)
        print("Checkpoint gespeichert:", args.save)


if __name__ == "__main__":
    main()
