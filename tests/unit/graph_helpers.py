import re


def assert_labels_balanced(graph: str) -> None:
    """Every filter output label must be consumed exactly once (FFmpeg errors otherwise)."""
    produced, consumed = [], []
    for chain in graph.split(";"):
        head = re.match(r"^((?:\[[^\]]+\])+)", chain)
        tail = re.search(r"((?:\[[^\]]+\])+)$", chain)
        ins = re.findall(r"\[([^\]]+)\]", head.group(1)) if head else []
        same = head and tail and tail.group(1) == head.group(1)
        outs = re.findall(r"\[([^\]]+)\]", tail.group(1)) if tail and not same else []
        consumed += [i for i in ins if not re.fullmatch(r"\d+:[av]", i)]
        produced += outs
    produced = [p for p in produced if p not in ("vout", "aout")]
    assert len(set(produced)) == len(produced), f"label produced twice: {sorted(produced)}"
    assert sorted(produced) == sorted(consumed), (sorted(produced), sorted(consumed))
