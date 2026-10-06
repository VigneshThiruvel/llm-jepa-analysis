#!/usr/bin/env python3
"""Exact language equivalence for NL-RX regexes (synth / turk targets).

NL-RX uses the brics-automaton dialect: `~` complement, `&` intersection, `|`, postfix
`* + ? {n} {n,} {n,m}`, character classes, `.`, plus `\\b`. Python's `re` cannot express
`~` or `&`, so this module decides equivalence with Brzozowski derivatives, which
support complement and intersection directly, and explores the product of the two
derivative automata (a DFA equivalence check without building the DFAs up front).

Semantics
  * Precedence (brics): `|` < `&` < concatenation < postfix < `~` (prefix, binds tightest,
    so `~(X)*` is `(~X)*`). Matching is whole-string, as in the original NL-RX eval.
  * Alphabet: printable ASCII plus one stand-in for every other character. The check
    runs over the minterms of the character sets in both regexes, so it is exact for
    all strings over that alphabet, not a sample.
  * `\\b` is a zero-width word boundary (word = [A-Za-z0-9_], string ends count as
    non-word), evaluated against the surrounding characters of the whole string, the
    way a backtracking engine evaluates it. With `boundary="eps"` it is treated as the
    empty string instead (robustness check: the result should not hinge on it).

API
  parse(s) -> AST (raises ParseError)
  equivalent(a, b, boundary="assert", max_states=200000) -> True / False / None (budget hit)
  fullmatch(ast, s, boundary=...) -> bool
"""
import string

WORD = frozenset(string.ascii_letters + string.digits + "_")
OTHER = "\x80"  # stands for every non-printable-ASCII character
UNIVERSE = frozenset(chr(i) for i in range(32, 127)) | {OTHER}


class ParseError(ValueError):
    pass


# ---------------------------------------------------------------- AST + smart constructors
# Nodes are hashable tuples: ("empty",) ("eps",) ("set", fs) ("bound",)
# ("cat", a, b) ("or", fs) ("and", fs) ("not", r) ("star", r)
EMPTY, EPS, BOUND = ("empty",), ("eps",), ("bound",)
ANY_STAR = None  # filled below


def cset(chars):
    chars = frozenset(chars)
    return ("set", chars) if chars else EMPTY


def star(r):
    if r in (EMPTY, EPS):
        return EPS
    if r[0] == "star":
        return r
    return ("star", r)


def neg(r):
    if r[0] == "not":
        return r[1]
    return ("not", r)


def cat(a, b):
    if a == EMPTY or b == EMPTY:
        return EMPTY
    if a == EPS:
        return b
    if b == EPS:
        return a
    if a[0] == "cat":  # right-associate
        return cat(a[1], cat(a[2], b))
    return ("cat", a, b)


def alt(*rs):
    items = set()
    for r in rs:
        if r[0] == "or":
            items |= r[1]
        elif r != EMPTY:
            items.add(r)
    if ANY_STAR in items:
        return ANY_STAR
    if not items:
        return EMPTY
    if len(items) == 1:
        return next(iter(items))
    return ("or", frozenset(items))


def conj(*rs):
    items = set()
    for r in rs:
        if r == EMPTY:
            return EMPTY
        if r[0] == "and":
            items |= r[1]
        elif r != ANY_STAR:
            items.add(r)
    if not items:
        return ANY_STAR
    if len(items) == 1:
        return next(iter(items))
    return ("and", frozenset(items))


ANY_STAR = ("star", ("set", UNIVERSE))


def repeat(r, lo, hi):
    out = EPS
    for _ in range(lo):
        out = cat(out, r)
    if hi is None:
        return cat(out, star(r))
    opt = EPS
    for _ in range(hi - lo):
        opt = alt(EPS, cat(r, opt))
    return cat(out, opt)


# ---------------------------------------------------------------- parser
class _P:
    def __init__(self, s):
        self.s, self.i = s, 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def eat(self, c=None):
        ch = self.peek()
        if ch is None or (c is not None and ch != c):
            raise ParseError(f"expected {c!r} at {self.i} in {self.s!r}")
        self.i += 1
        return ch

    def union(self):
        r = self.inter()
        while self.peek() == "|":
            self.eat()
            r = alt(r, self.inter())
        return r

    def inter(self):
        r = self.concat()
        while self.peek() == "&":
            self.eat()
            r = conj(r, self.concat())
        return r

    def concat(self):
        parts = []
        while self.peek() is not None and self.peek() not in "|&)":
            parts.append(self.repeat())
        out = EPS
        for p in reversed(parts):
            out = cat(p, out)
        return out

    def repeat(self):
        r = self.compl()
        while self.peek() in ("*", "+", "?", "{"):
            c = self.eat()
            if c == "*":
                r = star(r)
            elif c == "+":
                r = cat(r, star(r))
            elif c == "?":
                r = alt(EPS, r)
            else:
                j = self.s.find("}", self.i)
                if j < 0:
                    raise ParseError(f"unclosed {{ in {self.s!r}")
                body = self.s[self.i:j]
                self.i = j + 1
                try:
                    if "," in body:
                        lo, hi = body.split(",", 1)
                        lo, hi = int(lo), (int(hi) if hi.strip() else None)
                    else:
                        lo = hi = int(body)
                except ValueError:
                    raise ParseError(f"bad repeat {{{body}}} in {self.s!r}")
                if hi is not None and hi < lo or lo > 50 or (hi or 0) > 50:
                    raise ParseError(f"bad repeat {{{body}}}")
                r = repeat(r, lo, hi)
        return r

    def compl(self):
        if self.peek() == "~":
            self.eat()
            return neg(self.compl())
        return self.atom()

    def atom(self):
        c = self.peek()
        if c is None:
            raise ParseError(f"unexpected end in {self.s!r}")
        if c == "(":
            self.eat()
            r = self.union()
            self.eat(")")
            return r
        if c == "[":
            return self.cls()
        if c == ".":
            self.eat()
            return cset(UNIVERSE)
        if c == "\\":
            self.eat()
            e = self.eat()
            if e == "b":
                return BOUND
            return cset({e})
        if c in "*+?{}":
            raise ParseError(f"dangling {c!r} at {self.i} in {self.s!r}")
        self.eat()
        return cset({c})

    def cls(self):
        self.eat("[")
        negate = self.peek() == "^"
        if negate:
            self.eat()
        chars = set()
        first = True
        while self.peek() != "]" or first:
            first = False
            a = self.eat()
            if a == "\\":
                a = self.eat()
            if self.peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                self.eat("-")
                b = self.eat()
                if ord(b) < ord(a):
                    raise ParseError(f"bad range {a}-{b}")
                chars |= {chr(x) for x in range(ord(a), ord(b) + 1)}
            else:
                chars.add(a)
        self.eat("]")
        chars &= UNIVERSE
        return cset(UNIVERSE - chars if negate else chars)


def parse(s):
    p = _P(s)
    r = p.union()
    if p.i != len(s):
        raise ParseError(f"trailing input at {p.i} in {s!r}")
    return r


# ---------------------------------------------------------------- derivatives with \b context
def _eps(r, prev_w, next_w, bnd):
    """Does r match the empty string between a char of word-ness prev_w and one of next_w?"""
    t = r[0]
    if t in ("eps", "star"):
        return True
    if t in ("empty", "set"):
        return False
    if t == "bound":
        return True if bnd == "eps" else prev_w != next_w
    if t == "cat":
        return _eps(r[1], prev_w, next_w, bnd) and _eps(r[2], prev_w, next_w, bnd)
    if t == "or":
        return any(_eps(x, prev_w, next_w, bnd) for x in r[1])
    if t == "and":
        return all(_eps(x, prev_w, next_w, bnd) for x in r[1])
    if t == "not":
        return not _eps(r[1], prev_w, next_w, bnd)
    raise AssertionError(t)


def _deriv(r, c, prev_w, bnd, memo):
    key = (r, c, prev_w)
    if key in memo:
        return memo[key]
    t = r[0]
    nw = c in WORD
    if t == "set":
        out = EPS if c in r[1] else EMPTY
    elif t in ("eps", "empty", "bound"):
        out = EMPTY
    elif t == "cat":
        out = cat(_deriv(r[1], c, prev_w, bnd, memo), r[2])
        if _eps(r[1], prev_w, nw, bnd):
            out = alt(out, _deriv(r[2], c, prev_w, bnd, memo))
    elif t == "or":
        out = alt(*(_deriv(x, c, prev_w, bnd, memo) for x in r[1]))
    elif t == "and":
        out = conj(*(_deriv(x, c, prev_w, bnd, memo) for x in r[1]))
    elif t == "not":
        out = neg(_deriv(r[1], c, prev_w, bnd, memo))
    elif t == "star":
        out = cat(_deriv(r[1], c, prev_w, bnd, memo), r)
    else:
        raise AssertionError(t)
    memo[key] = out
    return out


def _sets(r, acc):
    t = r[0]
    if t == "set":
        acc.add(r[1])
    elif t == "cat":
        _sets(r[1], acc)
        _sets(r[2], acc)
    elif t in ("or", "and"):
        for x in r[1]:
            _sets(x, acc)
    elif t in ("not", "star"):
        _sets(r[1], acc)
    return acc


def _minterm_reps(*asts):
    """One representative char per class of characters no regex here can tell apart."""
    sets = set()
    for a in asts:
        _sets(a, sets)
    sets.add(WORD & UNIVERSE)
    sig = {}
    for ch in sorted(UNIVERSE):
        k = tuple(ch in s for s in sorted(sets, key=sorted))
        sig.setdefault(k, ch)
    return list(sig.values())


def fullmatch(r, s, boundary="assert"):
    memo, prev_w = {}, False
    for ch in s:
        ch = ch if ch in UNIVERSE else OTHER
        r = _deriv(r, ch, prev_w, boundary, memo)
        prev_w = ch in WORD
        if r == EMPTY:
            return False
    return _eps(r, prev_w, False, boundary)


def equivalent(a, b, boundary="assert", max_states=200000, witness=False):
    """True/False, or None if the explored product exceeded max_states.
    With witness=True returns (verdict, distinguishing string or None)."""
    if isinstance(a, str):
        a = parse(a)
    if isinstance(b, str):
        b = parse(b)
    reps = _minterm_reps(a, b)
    memo = {}
    start = (a, b, False)
    seen = {start: ""}
    frontier = [start]
    while frontier:
        nxt = []
        for st in frontier:
            ra, rb, pw = st
            if _eps(ra, pw, False, boundary) != _eps(rb, pw, False, boundary):
                return (False, seen[st]) if witness else False
            if ra == rb:
                continue
            for ch in reps:
                n = (_deriv(ra, ch, pw, boundary, memo), _deriv(rb, ch, pw, boundary, memo), ch in WORD)
                if n not in seen:
                    seen[n] = seen[st] + ch
                    if len(seen) > max_states:
                        return (None, None) if witness else None
                    nxt.append(n)
        frontier = nxt
    return (True, None) if witness else True
