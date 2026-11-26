#!/usr/bin/env python3
"""Writes tests/tiny/*.txt (inputs) and *.expected (hand-written expected dumps).

Expected results are written by hand below, NOT computed, so they check the
builders and the oracle alike. Dump format: "#docs\t<n>" then "term\tdf\tdocs"
in memcmp (byte) order. Re-run after editing: python3 tests/make_tiny.py
"""
from pathlib import Path

LONG = "x" * 10000

CASES = {
    # name: (input bytes, n_docs, [(term, [doc ids]), ...] in byte order)
    "basic": (
        b"the cat sat\nthe dog sat on the mat\ncat\n",
        3,
        [("cat", [0, 2]), ("dog", [1]), ("mat", [1]), ("on", [1]), ("sat", [0, 1]), ("the", [0, 1])],
    ),
    "empty": (b"", 0, []),
    "empty_docs": (b"\n\na b\n\n", 4, [("a", [2]), ("b", [2])]),
    "repeat": (b"x x x x\nx\ny x y\n", 3, [("x", [0, 1, 2]), ("y", [2])]),
    "prefix": (
        b"abc ab a\nab\nabcd a\n",
        3,
        [("a", [0, 2]), ("ab", [0, 1]), ("abc", [0]), ("abcd", [2])],
    ),
    "whitespace": (
        b"  lead\ttab  trail  \r\nmid\t\tdouble\x0b\x0cend\n   \n",
        3,
        [("double", [1]), ("end", [1]), ("lead", [0]), ("mid", [1]), ("tab", [0]), ("trail", [0])],
    ),
    "no_trailing_newline": (b"a b\nb c", 2, [("a", [0]), ("b", [0, 1]), ("c", [1])]),
    "utf8": (
        "café naïve\nnaïve 日本\n".encode(),
        2,
        [("café", [0]), ("naïve", [0, 1]), ("日本", [1])],
    ),
    "case": (b"B a\nb A\n", 2, [("A", [1]), ("B", [0]), ("a", [0]), ("b", [1])]),
    "punct": (b"a,b a , b\n", 1, [(",", [0]), ("a", [0]), ("a,b", [0]), ("b", [0])]),
    "all_docs": (b"z a\nz b\nz c\nz\n", 4, [("a", [0]), ("b", [1]), ("c", [2]), ("z", [0, 1, 2, 3])]),
    "single_doc": (b"only\n", 1, [("only", [0])]),
    "long_token": (
        f"{LONG} y\ny {LONG}z\n".encode(),
        2,
        [(LONG, [0]), (LONG + "z", [1]), ("y", [0, 1])],
    ),
}


def main():
    out = Path(__file__).resolve().parent / "tiny"
    out.mkdir(exist_ok=True)
    for name, (data, n_docs, terms) in CASES.items():
        assert [t for t, _ in terms] == sorted((t for t, _ in terms), key=str.encode), name
        (out / f"{name}.txt").write_bytes(data)
        lines = [f"#docs\t{n_docs}"] + [f"{t}\t{len(d)}\t{' '.join(map(str, d))}" for t, d in terms]
        (out / f"{name}.expected").write_bytes(("\n".join(lines) + "\n").encode())
    print(f"wrote {len(CASES)} cases to {out}")


if __name__ == "__main__":
    main()
