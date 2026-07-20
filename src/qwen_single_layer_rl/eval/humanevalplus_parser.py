from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass


_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_LINE_CODE_START = re.compile(
    r"^(?:@|async\s+def\s+|def\s+|class\s+|from\s+\S+\s+import\s+|import\s+\S+)",
)
_INLINE_DECLARATION = re.compile(r"(?:async\s+def|def|class)\s+[A-Za-z_]\w*")


@dataclass(frozen=True)
class ParserReceipt:
    strategy: str
    changed: bool
    raw_sha256: str
    parsed_sha256: str
    removed_prefix_chars: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_humanevalplus_prediction(text: str) -> tuple[str, ParserReceipt]:
    raw = text
    candidate = text
    strategy = "unchanged"

    fenced = _FENCED_CODE.findall(candidate)
    if fenced:
        candidate = fenced[0]
        strategy = "first_fenced_code_block"
    else:
        without_think = _THINK_BLOCK.sub("", candidate)
        if without_think != candidate:
            candidate = without_think
            strategy = "strip_complete_think_block"

        lines = candidate.splitlines(keepends=True)
        first_nonempty = next((index for index, line in enumerate(lines) if line.strip()), None)
        if first_nonempty is not None:
            first = lines[first_nonempty]
            stripped = first.lstrip()
            if not (_LINE_CODE_START.match(stripped) or first[:1].isspace()):
                code_start: tuple[int, int] | None = None
                for index in range(first_nonempty, len(lines)):
                    line = lines[index]
                    stripped_line = line.lstrip()
                    if _LINE_CODE_START.match(stripped_line):
                        code_start = (index, len(line) - len(stripped_line))
                        break
                    declaration = _INLINE_DECLARATION.search(line)
                    if declaration is not None:
                        code_start = (index, declaration.start())
                        break
                if code_start is None:
                    for index in range(first_nonempty + 1, len(lines)):
                        line = lines[index]
                        if line.strip() and line[:1].isspace():
                            code_start = (index, 0)
                            strategy = "strip_prefix_to_indented_function_body"
                            break
                if code_start is not None:
                    index, column = code_start
                    lines[index] = lines[index][column:]
                    candidate = "".join(lines[index:])
                    if strategy != "strip_prefix_to_indented_function_body":
                        strategy = "strip_prefix_to_python_declaration"

    receipt = ParserReceipt(
        strategy=strategy,
        changed=candidate != raw,
        raw_sha256=_sha256(raw),
        parsed_sha256=_sha256(candidate),
        removed_prefix_chars=max(0, len(raw) - len(candidate)),
    )
    return candidate, receipt


def install_evalscope_humanevalplus_parser() -> None:
    from evalscope.benchmarks.humanevalplus.humanevalplus_adapter import (
        HumanevalplusAdapter,
    )

    if getattr(HumanevalplusAdapter, "_qwen_parser_v2_installed", False):
        return

    def postprocess(cls: type, text: str) -> str:
        del cls
        parsed, _ = parse_humanevalplus_prediction(text)
        return parsed

    HumanevalplusAdapter._postprocess = classmethod(postprocess)
    HumanevalplusAdapter._qwen_parser_v2_installed = True
