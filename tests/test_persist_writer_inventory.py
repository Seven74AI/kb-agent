"""Enforce central durable-write choke point (agent.persist) and policy semantics."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from agent.persist import persist_durable_json, persist_durable_text

_REPO = Path(__file__).resolve().parents[1]

def _py_files_under(*roots: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        base = _REPO / root
        if not base.is_dir():
            continue
        out.extend(sorted(base.rglob("*.py")))
    for pattern in ("scripts/batch_*.py",):
        out.extend(sorted(_REPO.glob(pattern)))
    for rel in ("scripts/diarize.py", "scripts/transcribe.py"):
        path = _REPO / rel
        if path.is_file():
            out.append(path)
    return out


def _is_write_text_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
        return True
    return False


def _is_json_dump_to_disk(node: ast.Call) -> bool:
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "dump"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
    ):
        return False
    if not node.args:
        return False
    sink = node.args[-1]
    if isinstance(sink, ast.Name) and sink.id == "f":
        return True
    if isinstance(sink, ast.Attribute) and sink.attr in ("write", "writelines"):
        return True
    return False


def _collect_raw_disk_writers(tree: ast.AST, rel: str) -> list[str]:
    violations: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._func_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._func_stack.append(node.name)
            self.generic_visit(node)
            self._func_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._func_stack.append(node.name)
            self.generic_visit(node)
            self._func_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = self._func_stack[-1] if self._func_stack else "<module>"
            if _is_write_text_call(node):
                violations.append(f"{rel}:{func} -> write_text")
            elif _is_json_dump_to_disk(node):
                violations.append(f"{rel}:{func} -> json.dump")
            self.generic_visit(node)

    _Visitor().visit(tree)
    return violations


def test_inventory_no_raw_disk_writers_outside_persist_and_allowlist():
    errors: list[str] = []
    for path in _py_files_under("agent", "modal"):
        rel = str(path.relative_to(_REPO))
        if rel.replace("\\", "/") == "agent/persist.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        errors.extend(_collect_raw_disk_writers(tree, rel))
    for path in sorted(_REPO.glob("scripts/batch_*.py")):
        rel = str(path.relative_to(_REPO))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        errors.extend(_collect_raw_disk_writers(tree, rel))
    for rel in ("scripts/diarize.py", "scripts/transcribe.py"):
        path = _REPO / rel
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        errors.extend(_collect_raw_disk_writers(tree, rel))
    assert not errors, "raw durable writers outside persist:\n" + "\n".join(errors)


def _calls_name(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == name:
                return True
            if isinstance(child.func, ast.Attribute) and child.func.attr == name:
                return True
    return False


def _calls_persist_durable(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        if isinstance(target, ast.Attribute) and target.attr in (
            "persist_durable_text",
            "persist_durable_json",
        ):
            return True
        if isinstance(target, ast.Name) and target.id in (
            "persist_durable_text",
            "persist_durable_json",
        ):
            return True
    return False


def _persist_policy_kw(call: ast.Call) -> str | None:
    for kw in call.keywords:
        if kw.arg == "policy" and isinstance(kw.value, ast.Constant) and isinstance(
            kw.value.value,
            str,
        ):
            return kw.value.value
    return None


def _function_has_scrubbed_durable_write(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if _calls_name(func, "_persist_visible_error"):
        return True
    if _calls_name(func, "write_text") or _calls_name(func, "json.dump"):
        return True
    for child in ast.walk(func):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        name = None
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        if name not in ("persist_durable_text", "persist_durable_json"):
            continue
        policy = _persist_policy_kw(child)
        if policy in (None, "boundary4", "vault"):
            return True
    return False


def test_display_never_persists_path_scrub():
    errors: list[str] = []
    for path in _py_files_under("agent"):
        rel = str(path.relative_to(_REPO))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _calls_name(node, "sanitize_visible_message"):
                continue
            if not _function_has_scrubbed_durable_write(node):
                continue
            errors.append(f"{rel}:{node.name} sanitize_visible_message with durable write")
    assert not errors, "\n".join(errors)


def test_persist_boundary4_scrubs_url_and_credentials_preserves_path_and_length(tmp_path):
    long_tail = "z" * 5000
    raw = (
        f"see https://cdn.example.com/v.mp4?token=SECRET\n"
        f"api_key=supersecret\n"
        f"/var/kb-agent/data/artifact.json\n"
        f"{long_tail}"
    )
    out = tmp_path / "sidecar.txt"
    persist_durable_text(out, raw, policy="boundary4")
    written = out.read_text(encoding="utf-8")
    assert "token=SECRET" not in written
    assert "supersecret" not in written
    assert "[credential redacted]" in written
    assert "/var/kb-agent/data/artifact.json" in written
    assert "[path]" not in written
    assert len(written) > 4000


def test_persist_machine_local_writes_verbatim(tmp_path):
    raw = "api_key=keep sessionid=keep https://x.test/?q=1\n/root/secret/path\n"
    out = tmp_path / "local.conf"
    persist_durable_text(out, raw, policy="machine_local")
    assert out.read_text(encoding="utf-8") == raw


def test_persist_vault_scrubs_frontmatter_and_body_urls(tmp_path):
    tokenized = "https://cdn.example.com/v.mp4?token=SECRET"
    body = f"""---
source_url: {tokenized}
---

Body {tokenized}
"""
    out = tmp_path / "note.md"
    persist_durable_text(out, body, policy="vault")
    written = out.read_text(encoding="utf-8")
    assert "token=SECRET" not in written
    assert "https://cdn.example.com/v.mp4" in written


def test_scrub_durable_json_boundary4_applies_to_serialized_strings(tmp_path):
    payload = {"note": "api_key=leak https://a.test/?s=1"}
    path = tmp_path / "meta.json"
    persist_durable_json(path, payload, policy="boundary4")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert "leak" not in parsed["note"]
    assert "s=1" not in parsed["note"]
