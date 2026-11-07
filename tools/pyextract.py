#!/usr/bin/env python3
"""
pyextract - bundle a Python entry script plus the local code it really needs
into ONE minimal script.

Usage:
    python pyextract.py ENTRY.py [DIR ...] [-o OUT.py]

    ENTRY.py  the script to extract
    DIR ...   local search roots, e.g. the dir that contains the package
              "comp" for `from comp.encoder.vox import Foo`
    -o OUT    output file (default: stdout)

What it does
    * Parses ENTRY.py and keeps all of it (minus import statements).
    * Every name the code uses is resolved: local definition, local import
      (followed recursively through DIRs), or external import.
    * Only the top-level statements (functions, classes, enums, constants,
      variables, type aliases ...) that are actually referenced are copied.
    * External imports (stdlib / site-packages) are collected at the top.
    * Name clashes between modules are resolved by renaming (module_name).

Supported import styles
    from mod import part
    from mod import part as p
    from mod import (Part1, part2)
    import mod
    import pkg.sub.mod
    import pkg.sub.mod as alias        -> alias.Foo is rewritten to Foo
    from . import x / from .x import y  (relative)
    from mod import *                   (local modules)
    imports inside try/if blocks and inside functions (local from-imports)

Known limitations (static analysis only)
    * Comments / formatting are lost (code is regenerated with ast.unparse,
      Python 3.9+ needed).
    * Module-level side effects of non-entry modules (bare calls, registries,
      monkey patches) are not copied unless a needed name is bound by them.
    * A local module used as a plain value (e.g. passed as an argument) can
      not be inlined - a warning is printed.
    * `from m import x` of a variable that m rebinds at runtime via `global`
      keeps its import-time value (a snapshot copy is emitted). Writes
      like `m.x = 5` from other modules are not modelled.
    * Unresolvable local imports (e.g. optional modules in try/except) are
      kept unchanged, so the bundle behaves as before.
    * Names built dynamically (globals(), getattr, string annotations,
      __all__ tricks) are not tracked.
"""

import argparse
import ast
import builtins
import copy
import sys
from pathlib import Path

BUILTINS = set(dir(builtins))
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# --------------------------------------------------------------------------
# helpers: which names does a piece of code bind?
# --------------------------------------------------------------------------
def _collect(nodes):  # noqa: C901
    """Names bound by `nodes` (not descending into nested function/class
    bodies) and names declared global/nonlocal."""
    names, globs = set(), set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if isinstance(n, DEFS):
            names.add(n.name)
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            globs.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name != "*":
                    names.add((a.asname or a.name).split(".")[0])
        elif type(n).__name__ in ("MatchAs", "MatchStar") and getattr(n, "name", None):
            names.add(n.name)
        stack.extend(ast.iter_child_nodes(n))
    return names, globs


def collect_locals(node):
    """Local names of a function / lambda / class body."""
    names = set()
    if isinstance(node, FUNCS):
        a = node.args
        for x in a.posonlyargs + a.args + a.kwonlyargs:
            names.add(x.arg)
        for x in (a.vararg, a.kwarg):
            if x is not None:
                names.add(x.arg)
        body = node.body if isinstance(node.body, list) else [node.body]
    else:
        body = node.body
    found, globs = _collect(body)
    return (names | found) - globs


# --------------------------------------------------------------------------
# a parsed local module
# --------------------------------------------------------------------------
class Module:
    def __init__(self, ex, key, path, is_pkg):
        self.key = key
        self.path = path
        self.is_pkg = is_pkg
        self.package = key if is_pkg else key.rpartition(".")[0]
        self.tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=str(path))
        self.defs = {}  # name -> [top-level statements binding it]
        self.imports = {}  # local name -> import description
        self.modalias = {}  # dotted prefix used in code -> local module name
        self.stars = []  # local `from x import *`
        self.ext_stars = []  # external `from x import *`
        self.rebound = {  # names rebound at runtime via `global`
            n for node in ast.walk(self.tree) if isinstance(node, ast.Global) for n in node.names
        }
        self._scan(ex)

    def _scan(self, ex):
        for st in self.tree.body:
            if isinstance(st, ast.Import):
                for a in st.names:
                    self._import(ex, a)
            elif isinstance(st, ast.ImportFrom):
                self._from(ex, st)
            else:
                for n in _collect([st])[0]:
                    self.defs.setdefault(n, []).append(st)

    def _import(self, ex, a):
        if ex.is_local_module(a.name):
            if a.asname:
                self.imports[a.asname] = ("mod", a.name)
                self.modalias[a.asname] = a.name
            else:
                parts = a.name.split(".")
                for i in range(1, len(parts) + 1):
                    p = ".".join(parts[:i])
                    self.modalias[p] = p
                self.imports[parts[0]] = ("mod", parts[0])
        else:
            local = a.asname or a.name.split(".")[0]
            self.imports[local] = ("ext_import", a.name, a.asname)

    def _from(self, ex, st):
        mod = ex.relative(self, st.level, st.module)
        local = bool(st.level) or ex.is_local_module(mod)
        for a in st.names:
            if a.name == "*":
                (self.stars if local else self.ext_stars).append(mod)
                continue
            name = a.asname or a.name
            if local:
                self.imports[name] = ("from", mod, a.name)
            else:
                self.imports[name] = ("ext_from", mod, a.name)


# --------------------------------------------------------------------------
# AST rewriter: resolves every referenced name, pulls in what is needed
# --------------------------------------------------------------------------
class Rewriter(ast.NodeTransformer):
    def __init__(self, ex, mod):
        self.ex = ex
        self.mod = mod
        self.scopes = []  # list of (kind, names)

    def is_local(self, name):
        last = len(self.scopes) - 1
        for i in range(last, -1, -1):
            kind, names = self.scopes[i]
            if kind == "class" and i != last:
                continue
            if name in names:
                return True
        return False

    def _stmts(self, stmts):
        out = []
        for s in stmts:
            r = self.visit(s)
            if r is None:
                continue
            if isinstance(r, list):
                out.extend(r)
            else:
                out.append(r)
        return out or [ast.Pass()]

    # ---- scopes
    def _scoped(self, node):
        self.scopes.append(("func", collect_locals(node)))
        self.generic_visit(node)
        self.scopes.pop()
        return node

    visit_FunctionDef = _scoped
    visit_AsyncFunctionDef = _scoped
    visit_Lambda = _scoped

    def _comp(self, node):
        names = set()
        for g in node.generators:
            for n in ast.walk(g.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        self.scopes.append(("func", names))
        self.generic_visit(node)
        self.scopes.pop()
        return node

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def visit_ClassDef(self, node):
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [self.visit(k) for k in node.keywords]
        self.scopes.append(("class", collect_locals(node)))
        node.body = self._stmts(node.body)
        self.scopes.pop()
        return node

    # ---- names
    def visit_Name(self, node):
        if self.is_local(node.id):
            return node
        new = self.ex.need(self.mod, node.id)
        if new:
            node.id = new
        elif self.mod.imports.get(node.id, ("",))[0] == "mod":
            self.ex.warn(f"{self.mod.key}: local module '{node.id}' is used as a value; it cannot be inlined")
        return node

    def visit_Attribute(self, node):
        parts, cur = [], node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and not self.is_local(cur.id):
            parts.append(cur.id)
            parts.reverse()
            r = self.ex.resolve_attr(self.mod, parts)
            if r:
                final, used = r
                new = ast.Name(id=final, ctx=ast.Load())
                rest = parts[used:]
                for i, attr in enumerate(rest):
                    ctx = node.ctx if i == len(rest) - 1 else ast.Load()
                    new = ast.Attribute(value=new, attr=attr, ctx=ctx)
                if not rest:
                    new.ctx = node.ctx
                return new
        return self.generic_visit(node)

    def visit_Global(self, node):
        node.names = [self.ex.need(self.mod, n) or n for n in node.names]
        return node

    # ---- imports that live inside blocks / functions
    def visit_Import(self, node):
        for a in node.names:
            if self.ex.is_local_module(a.name):
                self.ex.warn(f"{self.mod.key}: nested local 'import {a.name}' cannot be inlined")
        return node

    def visit_ImportFrom(self, node):
        mod = self.ex.relative(self.mod, node.level, node.module)
        if not (node.level or self.ex.is_local_module(mod)):
            return node  # external import stays where it is
        assigns = []
        for a in node.names:
            if a.name == "*":
                self.ex.warn(f"{self.mod.key}: nested star import from '{mod}' skipped")
                continue
            src = self.ex.need_from(mod, a.name)
            if src is None:
                if self.ex.submodule_of(mod, a.name):
                    self.ex.warn(f"{self.mod.key}: nested import of local module '{mod}.{a.name}' cannot be inlined")
                    continue
                return node  # unresolvable (e.g. optional dependency): keep as is
            local = a.asname or a.name
            if not self.scopes:
                local = self.ex.need(self.mod, local) or local
            if local != src:
                assigns.append(
                    ast.Assign(
                        targets=[ast.Name(id=local, ctx=ast.Store())],
                        value=ast.Name(id=src, ctx=ast.Load()),
                    )
                )
        return assigns or [ast.Pass()]


# --------------------------------------------------------------------------
# the extractor
# --------------------------------------------------------------------------
class Extractor:
    def __init__(self, entry, roots):
        self.entry_path = Path(entry).resolve()
        self.roots = [Path(r).resolve() for r in roots]
        self.search = self.roots + [self.entry_path.parent]
        self.modules = {}
        self.find_cache = {}
        self.symbols = {}  # (module key, name) -> final name | None
        self.owner = {}  # final name -> symbol key
        self.ext_final = {}
        self.ext_list = []  # [(spec, final name)]
        self.star_mods = []
        self.future = []
        self.emitted = set()
        self.out = []
        self.warnings = []
        self.entry = self._load_entry()
        for n in self.entry.defs:  # entry names are never renamed
            self.owner[n] = (self.entry.key, n)

    # ---- module lookup
    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def _load_entry(self):
        key, is_pkg = None, False
        for root in self.roots:
            try:
                rel = self.entry_path.relative_to(root)
            except ValueError:
                continue
            parts = list(rel.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
                is_pkg = True
            key = ".".join(parts)
            break
        if not key:
            key = self.entry_path.stem
        m = Module(self, key, self.entry_path, is_pkg)
        self.modules[key] = m
        return m

    def is_local_module(self, dotted):
        top = (dotted or "").split(".")[0]
        if not top:
            return False
        return any((r / top).is_dir() or (r / (top + ".py")).is_file() for r in self.search)

    def find(self, dotted):
        """-> (path, is_pkg) ; is_pkg None means namespace dir; or None."""
        if not dotted:
            return None
        if dotted in self.find_cache:
            return self.find_cache[dotted]
        parts = dotted.split(".")
        res = None
        for root in self.search:
            base = root.joinpath(*parts)
            f = base.with_name(base.name + ".py")
            if f.is_file():
                res = (f, False)
                break
            if (base / "__init__.py").is_file():
                res = (base / "__init__.py", True)
                break
            if base.is_dir() and res is None:
                res = (base, None)
        self.find_cache[dotted] = res
        return res

    def get_module(self, dotted):
        if dotted in self.modules:
            return self.modules[dotted]
        r = self.find(dotted)
        m = None
        if r and r[1] is not None:
            m = Module(self, dotted, r[0], r[1])
        self.modules[dotted] = m
        return m

    def relative(self, M, level, module):
        if not level:
            return module or ""
        pkg = M.package.split(".") if M.package else []
        base = pkg[: max(0, len(pkg) - (level - 1))]
        return ".".join(base + ([module] if module else []))

    def submodule_of(self, T, o, depth=0):
        """Dotted name of submodule T.o if `o` (looked up in T) is a module."""
        Tm = self.get_module(T)
        if Tm is not None:
            if o in Tm.defs:
                return None
            imp = Tm.imports.get(o)
            if imp is not None:
                if imp[0] == "from" and (imp[1], imp[2]) != (T, o):
                    if depth < 10:
                        return self.submodule_of(imp[1], imp[2], depth + 1)
                    return None
                if imp[0] == "mod":
                    return imp[1]
                if imp[0] != "from":
                    return None
                # `from . import o` inside package T itself: o is a submodule
        return f"{T}.{o}" if self.find(f"{T}.{o}") else None

    # ---- naming
    def claim(self, sk, desired):
        n, i = desired, 1
        while n in self.owner and self.owner[n] != sk:
            i += 1
            n = f"{desired}_{i}"
        self.owner[n] = sk
        return n

    def alloc(self, M, name):
        sk = (M.key, name)
        if name in self.owner and self.owner[name] != sk:
            return self.claim(sk, f"{M.key.replace('.', '_')}__{name}")
        return self.claim(sk, name)

    def ext(self, spec, desired):
        key = (spec, desired)
        if key in self.ext_final:
            return self.ext_final[key]
        sk = ("ext", key)
        if spec[0] == "import" and "." in spec[1] and spec[2] is None:
            final = desired
            self.owner.setdefault(final, sk)
        else:
            final = self.claim(sk, desired)
        self.ext_final[key] = final
        self.ext_list.append((spec, final))
        return final

    # ---- name resolution
    def need(self, M, name):  # noqa: C901
        """Make sure `name` (as seen from module M) is available in the
        bundle. Returns its final name in the bundle, or None."""
        key = (M.key, name)
        if key in self.symbols:
            return self.symbols[key]
        self.symbols[key] = None  # cycle guard
        res = None
        if name in M.defs:
            res = self.alloc(M, name)
            self.symbols[key] = res
            for st in M.defs[name]:
                self.emit(M, st)
        elif name in M.imports:
            imp = M.imports[name]
            if imp[0] == "from":
                res = self.need_from(imp[1], imp[2])
                res = self._snapshot(M, name, imp, res)
            elif imp[0] == "ext_from":
                res = self.ext(("from", imp[1], imp[2]), name)
            elif imp[0] == "ext_import":
                res = self.ext(("import", imp[1], imp[2]), name)
            # "mod": module object, handled via attribute access
        else:
            for s in M.stars:
                T = self.get_module(s)
                if T is not None:
                    res = self.need(T, name)
                    if res:
                        break
            if res is None and M.ext_stars and name not in BUILTINS and not name.startswith("__"):
                for s in M.ext_stars:
                    if s not in self.star_mods:
                        self.star_mods.append(s)
        self.symbols[key] = res
        return res

    def _snapshot(self, M, name, imp, res):
        """`from m import x` copies the value of x at import time. If m
        rebinds x later (via `global`), a shared bundle variable would see
        the new value, so emit `name = x` right after x's definition."""
        if res is None:
            return res
        T = self.get_module(imp[1])
        if T is None or imp[2] not in T.defs or imp[2] not in T.rebound:
            return res
        if all(isinstance(st, DEFS) for st in T.defs[imp[2]]):
            return res  # functions / classes: rebinding is not a concern
        final = self.alloc(M, name)
        self.out.append(
            ast.Assign(
                targets=[ast.Name(id=final, ctx=ast.Store())],
                value=ast.Name(id=res, ctx=ast.Load()),
            )
        )
        return final

    def need_from(self, modname, orig):
        T = self.get_module(modname)
        if T is not None:
            r = self.need(T, orig)
            if r:
                return r
        if self.submodule_of(modname, orig):
            return None  # it is a submodule, handled by attribute access
        self.warn(f"cannot resolve '{orig}' from local module '{modname}'")
        return None

    def resolve_attr(self, M, parts):  # noqa: C901
        """parts = ['alias', 'Foo', 'bar'] -> (final name of Foo, 2) if
        'alias' is a local module, else None."""
        found = None
        for i in range(len(parts), 0, -1):
            k = ".".join(parts[:i])
            if k in M.modalias:
                found = (M.modalias[k], i)
                break
        if not found:
            imp = M.imports.get(parts[0])
            if imp and imp[0] == "from":
                sub = self.submodule_of(imp[1], imp[2])
                if sub:
                    found = (sub, 1)
        if not found:
            return None
        T, i = found
        while i < len(parts):
            sub = self.submodule_of(T, parts[i])
            if not sub:
                break
            T = sub
            i += 1
        if i >= len(parts):
            return None
        Tm = self.get_module(T)
        if Tm is None:
            return None
        final = self.need(Tm, parts[i])
        if final is None:
            self.warn(f"cannot resolve '{parts[i]}' in local module '{T}'")
            return None
        return final, i + 1

    # ---- emission (dependencies are emitted before their users)
    def emit(self, M, st):
        if id(st) in self.emitted:
            return
        self.emitted.add(id(st))
        node = copy.deepcopy(st)
        if isinstance(node, DEFS):
            node.name = self.need(M, node.name) or node.name
        node = Rewriter(self, M).visit(node)
        for n in node if isinstance(node, list) else [node]:
            if n is not None:
                self.out.append(n)

    # ---- driver
    def run(self):
        body = self.entry.tree.body
        doc = None
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0].value, "value", None), str):
            doc = body[0]
            self.emitted.add(id(doc))
        for st in body:
            if isinstance(st, ast.ImportFrom) and st.module == "__future__":
                self.future.extend(a.name for a in st.names if a.name not in self.future)
                continue
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                continue
            self.emit(self.entry, st)

        mod = ast.Module(body=self.out, type_ignores=[])
        ast.fix_missing_locations(mod)
        code = ast.unparse(mod)

        head = []
        if doc is not None:
            text = doc.value.value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            head.append('"""' + text + '"""')
        if self.future:
            head.append("from __future__ import " + ", ".join(self.future))
        imports = self._import_lines()
        if imports:
            head.append("\n".join(imports))
        return "\n\n".join(head + [code]) + "\n"

    def _import_lines(self):
        plain, froms = [], {}
        for spec, final in self.ext_list:
            if spec[0] == "import":
                _, mod, asname = spec
                if asname is None and "." in mod or final == mod:
                    plain.append(f"import {mod}")
                else:
                    plain.append(f"import {mod} as {final}")
            else:
                _, mod, orig = spec
                item = orig if orig == final else f"{orig} as {final}"
                froms.setdefault(mod, [])
                if item not in froms[mod]:
                    froms[mod].append(item)
        lines = sorted(set(plain))
        for mod in sorted(froms):
            lines.append(f"from {mod} import {', '.join(froms[mod])}")
        for s in self.star_mods:
            lines.append(f"from {s} import *")
        return lines


def extract(entry, dirs=()):
    """Return the bundled source code as a string."""
    sys.setrecursionlimit(10000)
    ex = Extractor(entry, dirs)
    code = ex.run()
    for w in ex.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bundle a script and the local code it needs into one minimal script.")
    ap.add_argument("entry", help="entry script (.py)")
    ap.add_argument("dirs", nargs="*", help="local search roots (e.g. the dir containing package 'comp')")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    args = ap.parse_args(argv)
    code = extract(args.entry, args.dirs)
    if args.output:
        Path(args.output).write_text(code, encoding="utf-8")
        print(f"written: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    main()
