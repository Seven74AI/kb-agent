"""Enforce central durable-write choke point (agent.persist) and policy semantics."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from agent.persist import persist_durable_json, persist_durable_text

_REPO = Path(__file__).resolve().parents[1]

# Dev/benchmark scripts that may write non-vault artifacts (reports, fixtures) without persist choke.
_SCRIPTS_RAW_WRITE_ALLOWLIST = frozenset(
    {
        "scripts/note_quality_ab_benchmark.py",
        "scripts/regenerate_detect_type_benchmark_fixture.py",
    }
)

# machine_local operator-secret cookie upload, byte-exact by design
_AGENT_WRITE_BYTES_ALLOWLIST = frozenset({"agent/webhook.py"})

# Ephemeral /tmp byte staging inside Modal GPU workers (not durable kb artifacts)
_OPEN_WRITE_ALLOWLIST = frozenset(
    {
        "modal/kb_transcribe.py",
        "modal/kb_ocr.py",
    },
)


def _py_files_under(*roots: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        base = _REPO / root
        if not base.is_dir():
            continue
        out.extend(sorted(base.rglob("*.py")))
    return out


def _inventory_script_paths() -> list[Path]:
    return sorted(_REPO.glob("scripts/*.py"))


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


def _is_json_dumps_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dumps"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
    )


def _expr_contains_json_dumps(expr: ast.AST) -> bool:
    for child in ast.walk(expr):
        if _is_json_dumps_call(child):
            return True
    return False


def _is_json_dumps_write(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in ("write", "writelines"):
        return False
    if not node.args:
        return False
    return _expr_contains_json_dumps(node.args[0])


def _is_path_write_bytes(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "write_bytes"):
        return False
    return isinstance(node.func.value, (ast.Name, ast.Attribute))


_NON_BUILTIN_OPEN_MODULES = frozenset(
    {
        "gzip",
        "bz2",
        "lzma",
        "zipfile",
        "wave",
        "codecs",
        "io",
        "Image",
        "pymupdf",
        "fitz",
        "PIL",
    },
)


def _is_open_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "open":
        return True
    if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in _NON_BUILTIN_OPEN_MODULES:
            return False
        return True
    return False


def _open_mode_from_call(node: ast.Call) -> str:
    mode = "r"
    if len(node.args) >= 2:
        arg = node.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            mode = arg.value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(
            kw.value.value,
            str,
        ):
            mode = kw.value.value
    return mode


def _is_write_open_mode(mode: str) -> bool:
    if not mode or mode == "r":
        return False
    return mode[0] in ("w", "a", "x")


def _with_handle_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple):
        return {
            elt.id
            for elt in node.elts
            if isinstance(elt, ast.Name)
        }
    return set()


def _block_writes_handles(stmts: list[ast.stmt], handle_names: set[str]) -> bool:
    if not handle_names:
        return False
    for stmt in stmts:
        for child in ast.walk(stmt):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            if child.func.attr not in ("write", "writelines"):
                continue
            if isinstance(child.func.value, ast.Name) and child.func.value.id in handle_names:
                return True
    return False


def _collect_raw_disk_writers(
    tree: ast.AST,
    rel: str,
    *,
    allow_write_bytes: bool = False,
    allow_open_write: bool = False,
) -> list[str]:
    violations: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._func_stack: list[str] = []
            self._allow_write_bytes = allow_write_bytes
            self._allow_open_write = allow_open_write

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._func_stack.append(node.name)
            self.generic_visit(node)
            self._func_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._func_stack.append(node.name)
            self.generic_visit(node)
            self._func_stack.pop()

        def visit_With(self, node: ast.With) -> None:
            if not self._allow_open_write:
                for item in node.items:
                    ctx = item.context_expr
                    if not isinstance(ctx, ast.Call) or not _is_open_call(ctx):
                        continue
                    if not _is_write_open_mode(_open_mode_from_call(ctx)):
                        continue
                    names = _with_handle_names(item.optional_vars)
                    if _block_writes_handles(node.body, names):
                        func = self._func_stack[-1] if self._func_stack else "<module>"
                        violations.append(f"{rel}:{func} -> open+write")
                        break
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            func = self._func_stack[-1] if self._func_stack else "<module>"
            if _is_write_text_call(node):
                violations.append(f"{rel}:{func} -> write_text")
            elif _is_json_dump_to_disk(node):
                violations.append(f"{rel}:{func} -> json.dump")
            elif _is_json_dumps_write(node):
                violations.append(f"{rel}:{func} -> json.dumps+write")
            elif _is_path_write_bytes(node) and not self._allow_write_bytes:
                violations.append(f"{rel}:{func} -> write_bytes")
            self.generic_visit(node)

    _Visitor().visit(tree)
    return violations


def test_inventory_no_raw_disk_writers_outside_persist_and_allowlist():
    errors: list[str] = []
    for path in _py_files_under("agent", "modal"):
        rel = str(path.relative_to(_REPO)).replace("\\", "/")
        if rel == "agent/persist.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        allow_wb = rel in _AGENT_WRITE_BYTES_ALLOWLIST
        allow_open = rel in _OPEN_WRITE_ALLOWLIST
        errors.extend(
            _collect_raw_disk_writers(
                tree,
                rel,
                allow_write_bytes=allow_wb,
                allow_open_write=allow_open,
            ),
        )
    for path in _inventory_script_paths():
        rel = str(path.relative_to(_REPO)).replace("\\", "/")
        if rel in _SCRIPTS_RAW_WRITE_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        errors.extend(_collect_raw_disk_writers(tree, rel))
    assert not errors, "raw durable writers outside persist:\n" + "\n".join(errors)


def test_collect_raw_disk_writers_flags_path_write_text():
    tree = ast.parse("def main():\n    note.write_text(body, encoding='utf-8')\n")
    hits = _collect_raw_disk_writers(tree, "example.py")
    assert hits == ["example.py:main -> write_text"]


def test_collect_raw_disk_writers_flags_json_dump_to_open_handle():
    src = (
        "import json\n"
        "def main():\n"
        "    with open(path, 'w') as f:\n"
        "        json.dump(payload, f)\n"
    )
    tree = ast.parse(src)
    hits = _collect_raw_disk_writers(tree, "example.py")
    assert hits == ["example.py:main -> json.dump"]


def test_collect_raw_disk_writers_flags_bare_open_write():
    src = "def main():\n    with open(path, 'w') as f:\n        f.write('payload')\n"
    tree = ast.parse(src)
    hits = _collect_raw_disk_writers(tree, "scripts/evil.py")
    assert hits == ["scripts/evil.py:main -> open+write"]


def test_collect_raw_disk_writers_flags_bare_open_writelines():
    src = "def main():\n    with open(path, 'a') as f:\n        f.writelines(lines)\n"
    tree = ast.parse(src)
    hits = _collect_raw_disk_writers(tree, "example.py")
    assert hits == ["example.py:main -> open+write"]


def test_collect_raw_disk_writers_flags_json_dumps_write():
    src = (
        "import json\n"
        "def main():\n"
        "    f.write(json.dumps(safe, ensure_ascii=False) + '\\n')\n"
    )
    tree = ast.parse(src)
    hits = _collect_raw_disk_writers(tree, "example.py")
    assert hits == ["example.py:main -> json.dumps+write"]


def test_collect_raw_disk_writers_flags_path_write_bytes():
    src = "def upload():\n    dest_path.write_bytes(payload)\n"
    tree = ast.parse(src)
    hits = _collect_raw_disk_writers(tree, "scripts/evil.py")
    assert hits == ["scripts/evil.py:upload -> write_bytes"]


def test_collect_raw_disk_writers_ignores_allowlisted_write_bytes():
    src = "def upload_cookies():\n    dest_path.write_bytes(payload)\n"
    tree = ast.parse(src)
    hits = _collect_raw_disk_writers(
        tree,
        "agent/webhook.py",
        allow_write_bytes=True,
    )
    assert hits == []


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


def test_persist_machine_local_append_preserves_existing(tmp_path):
    out = tmp_path / "batch.log"
    persist_durable_text(out, "line1\n", policy="machine_local")
    persist_durable_text(out, "line2\n", policy="machine_local", append=True)
    assert out.read_text(encoding="utf-8") == "line1\nline2\n"


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
