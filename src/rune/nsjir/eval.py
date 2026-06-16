"""Reference evaluator for NSJIR primitive terms."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from rune.nsjir.terms import Term


def evaluate(term: Term, env: Mapping[str, Any] | None = None) -> Any:
    env = env or {}
    values = [evaluate(arg, env) for arg in term.args]

    match term.op:
        case "const":
            return term.attrs["value"]
        case "var":
            return env[term.attrs["name"]]
        case "add":
            return values[0] + values[1]
        case "sub":
            return values[0] - values[1]
        case "mul":
            return values[0] * values[1]
        case "mod":
            return values[0] % term.attrs["modulus"]
        case "mod_add":
            return (values[0] + values[1]) % term.attrs["modulus"]
        case "crt":
            return _crt(values[0], values[1])
        case "lookup":
            return values[0][values[1]]
        case "copy":
            return values[0]
        case "gather":
            sequence, indices = values
            return [sequence[index] for index in indices]
        case "sort":
            return sorted(values[0])
        case "topk":
            return sorted(values[0], reverse=True)[: term.attrs["k"]]
        case "regex_match" | "wfa_match":
            return re.fullmatch(term.attrs["pattern"], values[0]) is not None
        case "rewrite":
            return str(values[0]).replace(term.attrs["old"], term.attrs["new"])
        case "fold":
            return _fold(term.attrs["fn"], values[0], term.attrs["initial"])
        case "scan":
            return _scan(term.attrs["fn"], values[0], term.attrs["initial"])
        case "dp_fibonacci":
            return _fibonacci(values[0])
        case "bfs":
            return _bfs(values[0], values[1])
        case "dfs":
            return _dfs(values[0], values[1])
        case "fix":
            return values[0]
        case _:
            raise ValueError(f"Unsupported NSJIR op: {term.op}")


def _fold(fn: str, sequence: list[Any], initial: Any) -> Any:
    current = initial
    for item in sequence:
        if fn == "add":
            current += item
        elif fn == "mul":
            current *= item
        else:
            raise ValueError(f"Unsupported fold fn: {fn}")
    return current


def _scan(fn: str, sequence: list[Any], initial: Any) -> list[Any]:
    current = initial
    output = []
    for item in sequence:
        if fn == "add":
            current += item
        elif fn == "mul":
            current *= item
        else:
            raise ValueError(f"Unsupported scan fn: {fn}")
        output.append(current)
    return output


def _fibonacci(count: int) -> int:
    previous, current = 0, 1
    for _ in range(count):
        previous, current = current, previous + current
    return previous


def _crt(residues: list[int], moduli: list[int]) -> int:
    modulus_product = 1
    for modulus in moduli:
        modulus_product *= modulus

    total = 0
    for residue, modulus in zip(residues, moduli, strict=True):
        partial = modulus_product // modulus
        total += residue * partial * pow(partial, -1, modulus)
    return total % modulus_product


def _bfs(graph: dict[Any, list[Any]], start: Any) -> list[Any]:
    seen = {start}
    queue = [start]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return order


def _dfs(graph: dict[Any, list[Any]], start: Any) -> list[Any]:
    seen: set[Any] = set()
    order: list[Any] = []

    def visit(node: Any) -> None:
        if node in seen:
            return
        seen.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            visit(neighbor)

    visit(start)
    return order