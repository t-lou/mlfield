"""
BERT-Entwurf in reinem PyTorch (ohne nn.MultiheadAttention / nn.TransformerEncoder).

Ziel: Self-Attention und den BERT-Encoder wirklich verstehen und per
Masked Language Modeling (MLM) trainieren.

Nutzung:
    python bert.py --text corpus.txt          # eigenes Textkorpus (ein Satz/Absatz pro Zeile)
    python bert.py --smoke                    # Schnelltest mit synthetischen Daten

Vereinfachungen gegenüber dem Original (Devlin et al., 2018):
  - Wort-Tokenizer statt WordPiece (leicht austauschbar, siehe Tokenizer)
  - nur MLM, kein Next Sentence Prediction (NSP gilt ohnehin als verzichtbar, vgl. RoBERTa)
  - kleine Modellgröße, damit Training auf CPU/Consumer-GPU möglich ist
"""

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIALS = [PAD, UNK, CLS, SEP, MASK]


# ----------------------------------------------------------------------------
# 1. Tokenizer (bewusst simpel)
# ----------------------------------------------------------------------------
class Tokenizer:
    """Wort-Tokenizer. Für echtes BERT-Verhalten durch WordPiece ersetzen
    (z. B. HuggingFace `tokenizers`), Schnittstelle bleibt gleich."""

    def __init__(self, lines, max_vocab=8000):
        counts = Counter(w for line in lines for w in line.lower().split())
        words = [w for w, _ in counts.most_common(max_vocab - len(SPECIALS))]
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id = self.stoi[PAD], self.stoi[UNK]
        self.cls_id, self.sep_id, self.mask_id = self.stoi[CLS], self.stoi[SEP], self.stoi[MASK]

    def __len__(self):
        return len(self.itos)

    def encode(self, line):
        return [self.stoi.get(w, self.unk_id) for w in line.lower().split()]


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

    Form-Notation: B=Batch, T=Sequenzlänge, H=Köpfe, D=d_model, d=D/H (Kopfdimension)
    """

    def __init__(self, cfg: BertConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.d = cfg.d_model // cfg.n_heads
        # Eine Matrix für Q, K, V gemeinsam (äquivalent zu drei getrennten Linears)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask, return_attn=False):
        # x: (B, T, D); key_padding_mask: (B, T), True = echtes Token, False = Padding
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # je (B, T, D)
        # In Köpfe aufteilen: (B, T, D) -> (B, H, T, d)
        q, k, v = (t.view(B, T, self.h, self.d).transpose(1, 2) for t in (q, k, v))

        # Ähnlichkeit jedes Query mit jedem Key, skaliert mit sqrt(d)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)  # (B, H, T, T)
        # Padding-Positionen als Keys ausblenden (BERT ist bidirektional: keine kausale Maske!)
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)  # Zeilen summieren zu 1
        out = self.drop(attn) @ v  # (B, H, T, d) gewichtete Summe der Values
        out = out.transpose(1, 2).reshape(B, T, D)  # Köpfe wieder zusammenführen
        out = self.out(out)
        return (out, attn) if return_attn else out


class EncoderLayer(nn.Module):
    """Post-LN wie im Original-BERT: x = LN(x + Sublayer(x))."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.attn = SelfAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=1e-12)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
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
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)  # gelernt, nicht sinusförmig
        self.seg_emb = nn.Embedding(2, cfg.d_model)  # Segment A/B (hier nur für Vollständigkeit)
        self.emb_ln = nn.LayerNorm(cfg.d_model, eps=1e-12)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))

        # MLM-Kopf: Transform + Projektion auf Vokabular, Gewichte mit Embedding geteilt (weight tying)
        self.mlm_dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlm_ln = nn.LayerNorm(cfg.d_model, eps=1e-12)
        self.mlm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

        self.apply(self._init)

    @staticmethod
    def _init(m):
        # BERT-Init: Normalverteilung mit std=0.02
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, ids, segments=None, return_attn=False):
        B, T = ids.shape
        mask = ids != self.cfg.pad_id  # (B, T)
        pos = torch.arange(T, device=ids.device)[None, :]
        if segments is None:
            segments = torch.zeros_like(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos) + self.seg_emb(segments)
        x = self.emb_drop(self.emb_ln(x))
        attns = []
        for layer in self.layers:
            if return_attn:
                x, a = layer(x, mask, return_attn=True)
                attns.append(a)
            else:
                x = layer(x, mask)
        return (x, attns) if return_attn else x

    def forward(self, ids, segments=None):
        h = self.encode(ids, segments)  # (B, T, D)
        h = self.mlm_ln(F.gelu(self.mlm_dense(h)))
        return h @ self.tok_emb.weight.T + self.mlm_bias  # Logits: (B, T, V)


# ----------------------------------------------------------------------------
# 3. Daten + MLM-Masking
# ----------------------------------------------------------------------------
def mlm_mask(ids, tok: Tokenizer, p=0.15):
    """BERT-Masking: 15 % der Tokens werden ausgewählt; davon
    80 % -> [MASK], 10 % -> zufälliges Token, 10 % -> unverändert.
    Labels sind -100 (ignoriert) für alle nicht ausgewählten Positionen."""
    ids = ids.clone()
    labels = ids.clone()
    special = (ids == tok.pad_id) | (ids == tok.cls_id) | (ids == tok.sep_id)
    chosen = (torch.rand(ids.shape) < p) & ~special
    labels[~chosen] = -100

    r = torch.rand(ids.shape)
    ids[chosen & (r < 0.8)] = tok.mask_id
    rand_pos = chosen & (r >= 0.8) & (r < 0.9)
    ids[rand_pos] = torch.randint(len(SPECIALS), len(tok), (int(rand_pos.sum()),))
    return ids, labels


class TextDataset(Dataset):
    """Jede Zeile wird zu [CLS] tokens [SEP], auf max_len gekürzt und gepaddet."""

    def __init__(self, lines, tok: Tokenizer, max_len):
        self.items = []
        for line in lines:
            body = tok.encode(line)[: max_len - 2]
            if len(body) < 3:
                continue
            seq = [tok.cls_id] + body + [tok.sep_id]
            seq += [tok.pad_id] * (max_len - len(seq))
            self.items.append(torch.tensor(seq))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# ----------------------------------------------------------------------------
# 4. Training
# ----------------------------------------------------------------------------
def train(model, loader, tok, epochs, lr, device, warmup_frac=0.1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    warmup = max(1, int(warmup_frac * total))
    # linearer Warmup, dann linearer Abfall (wie im BERT-Paper)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / (total - warmup))
    )
    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            inp, labels = mlm_mask(batch, tok)
            inp, labels = inp.to(device), labels.to(device)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {loss.item():.3f} ppl {math.exp(loss.item()):.1f}")


@torch.no_grad()
def fill_mask(model, tok, sentence, device, topk=5):
    """Sanity-Check: Satz mit [MASK] eingeben, Top-k-Vorhersagen ansehen."""
    model.eval()
    ids = (
        [tok.cls_id]
        + [tok.stoi.get(w.lower(), tok.unk_id) if w != MASK else tok.mask_id for w in sentence.split()]
        + [tok.sep_id]
    )
    x = torch.tensor([ids], device=device)
    probs = model(x)[0].softmax(-1)
    for i, t in enumerate(ids):
        if t == tok.mask_id:
            top = probs[i].topk(topk)
            print(f"Position {i}:", [(tok.itos[j], round(p.item(), 3)) for p, j in zip(top.values, top.indices)])


def synthetic_corpus(n=3000, seed=0):
    """Kleine Grammatik mit festen Mustern, damit MLM-Loss im Schnelltest klar fällt."""
    rng = random.Random(seed)
    subj = ["the cat", "the dog", "a bird", "my friend", "the teacher"]
    verb = ["sees", "likes", "chases", "finds", "helps"]
    obj = ["the ball", "a tree", "the car", "a book", "the house"]
    return [f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)}" for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, help="Textdatei, ein Satz/Absatz pro Zeile")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke or not args.text:
        lines = synthetic_corpus()
    else:
        with open(args.text, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

    tok = Tokenizer(lines)
    cfg = BertConfig(vocab_size=len(tok), max_len=args.max_len, pad_id=tok.pad_id)
    if args.smoke:
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff = 64, 4, 2, 256

    model = Bert(cfg).to(device)
    print(f"Parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, Vokabular: {len(tok)}")
    loader = DataLoader(TextDataset(lines, tok, cfg.max_len), batch_size=args.batch, shuffle=True)
    train(model, loader, tok, args.epochs, args.lr, device)
    fill_mask(model, tok, "the cat [MASK] the ball", device)


if __name__ == "__main__":
    main()
