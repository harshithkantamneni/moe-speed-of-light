"""Short table captions: keep a generated table's first caption sentence as its caption and move the rest into a
note set below the table (\\tabnote, defined in paper/macros.tex). Called by the scripts that write the tables; can
also be run by hand on a table file.

    python scripts/tabnote.py paper/tab_dm.tex
"""
import sys


def _caption_span(s):
    i = s.index("\\caption{") + len("\\caption{")
    d, j = 1, i
    while d:
        d += (s[j] == "{") - (s[j] == "}")
        j += 1
    return i, j - 1   # the caption's text is s[i:j-1]


def _first_sentence_end(cap):
    d = 0
    for k, ch in enumerate(cap):
        d += (ch == "{") - (ch == "}")
        if ch == "." and d == 0 and k + 1 < len(cap) and cap[k + 1] in " \n":
            rest = cap[k + 1:].lstrip()
            if rest and (rest[0].isupper() or rest[0] in "\\$"):
                return k + 1
    return None


def split_caption(path):
    s = open(path).read()
    if "\\tabnote{" in s:
        return False
    i, j = _caption_span(s)
    cap = s[i:j]
    k = _first_sentence_end(cap)
    if k is None:
        return False
    title, note = cap[:k].strip(), cap[k:].strip()
    s = s[:i] + title + s[j:]
    end = s.rindex("\\end{table")
    s = s[:end] + "\\tabnote{" + note + "}" + s[end:]
    open(path, "w").write(s)
    return True


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, split_caption(f))
