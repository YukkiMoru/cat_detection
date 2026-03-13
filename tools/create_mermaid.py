#!/usr/bin/env python
# uv run tools/create_mermaid.py
# src/* or tools/* -> custom_diagram.mmd

import ast
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README_PATH = PROJECT_ROOT / "README.md"
MERMAID_PATH = PROJECT_ROOT / "custom_diagram.mmd"
SVG_PATH = PROJECT_ROOT / "docs" / "custom_diagram.svg"
TARGET_DIRS = ("src", "tools")
README_START = "<!-- AUTO-GENERATED-DIAGRAM:START -->"
README_END = "<!-- AUTO-GENERATED-DIAGRAM:END -->"


def read_python_tree(py_file: Path) -> ast.AST:
    return ast.parse(py_file.read_text(encoding="utf-8"))


def collect_import_map(tree: ast.AST) -> dict[str, str]:
    import_map: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            target_module = node.module.split(".")[-1]
            for imported_name in node.names:
                import_map[imported_name.name] = target_module
        elif isinstance(node, ast.Import):
            for imported_name in node.names:
                target_module = imported_name.name.split(".")[-1]
                import_map[imported_name.name] = target_module
    return import_map


def collect_class_members(tree: ast.AST) -> list[str]:
    members: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = ", ".join(arg.arg for arg in node.args.args)
            members.append(f"            +{node.name}({args})")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    members.append(f"            +{target.id}")
    return members


def detect_folder_refs(tree: ast.AST, module_name: str) -> set[tuple[str, str]]:
    folder_refs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.lower()
            if "dataset" in value:
                folder_refs.add((module_name, "dataset"))
            if "models/" in value or ".pt" in value or ".onnx" in value:
                folder_refs.add((module_name, "models"))
    return folder_refs


def build_call_label(node: ast.Call, func_name: str) -> str:
    args_str = "..."
    if hasattr(ast, "unparse"):
        rendered_args = [ast.unparse(arg).replace('"', "'") for arg in node.args]
        args_str = ", ".join(rendered_args)
        if len(args_str) > 30:
            args_str = args_str[:27] + "..."
    return f"{func_name}({args_str})"


def collect_relationships(
    tree: ast.AST,
    module_name: str,
    import_map: dict[str, str],
) -> dict[tuple[str, str], set[str]]:
    relationships: dict[tuple[str, str], set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = ""
        target_module = ""

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            target_module = import_map.get(func_name, "")
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                target_module = import_map.get(node.func.value.id, "")

        if not target_module:
            continue

        rel_key = (module_name, target_module)
        relationships.setdefault(rel_key, set()).add(build_call_label(node, func_name))
    return relationships


def style_line(namespace_name: str, module_name: str) -> str | None:
    if module_name == "inference":
        return f"    style {module_name} core"
    if namespace_name == "tools":
        return f"    style {module_name} tool"
    if module_name == "main":
        return f"    style {module_name} main_logic"
    return None


def render_namespace(
    namespace_name: str,
    modules_found: list[str],
    relationships: dict[tuple[str, str], set[str]],
    folder_refs: set[tuple[str, str]],
    styles: list[str],
) -> list[str]:
    lines = [f"    namespace {namespace_name} {{"]
    dir_path = PROJECT_ROOT / namespace_name

    for py_file in sorted(dir_path.glob("*.py")):
        if py_file.name == "__init__.py":
            continue

        module_name = py_file.stem
        modules_found.append(module_name)
        lines.append(f"        class {module_name} {{")
        lines.append("            <<Module>>")

        try:
            tree = read_python_tree(py_file)
            import_map = collect_import_map(tree)
            lines.extend(collect_class_members(tree))
            folder_refs.update(detect_folder_refs(tree, module_name))

            module_relationships = collect_relationships(tree, module_name, import_map)
            for rel_key, labels in module_relationships.items():
                relationships.setdefault(rel_key, set()).update(labels)
        except Exception as exc:
            print(f"解析エラー: {py_file}: {exc}")

        lines.append("        }")

        current_style = style_line(namespace_name, module_name)
        if current_style:
            styles.append(current_style)

    lines.append("    }")
    return lines


def build_mermaid_text() -> str:
    mermaid_lines = [
        "classDiagram",
        "    %% スタイルの定義",
        "    classDef core fill:#f96,stroke:#333,stroke-width:2px;  %% 推論（中心）",
        "    classDef tool fill:#dfd,stroke:#333;                 %% ツール系",
        "    classDef main_logic fill:#cef,stroke:#333;           %% メイン実行系",
        "    classDef directory fill:#eee,stroke:#666,stroke-dasharray: 5 5; %% フォルダ系",
        "    classDef external fill:#5865F2,stroke:#fff,color:#fff,stroke-width:2px; %% 外部サービス（Discord）",  # ★追加
    ]
    styles: list[str] = []
    modules_found: list[str] = []
    relationships: dict[tuple[str, str], set[str]] = {}
    folder_refs: set[tuple[str, str]] = set()

    for target_dir in TARGET_DIRS:
        dir_path = PROJECT_ROOT / target_dir
        if dir_path.exists():
            mermaid_lines.extend(
                render_namespace(
                    target_dir,
                    modules_found,
                    relationships,
                    folder_refs,
                    styles,
                )
            )

    mermaid_lines.extend(
        [
            "    namespace directories {",
            "        class dataset {",
            "            <<Folder>>",
            "        }",
            "        class models {",
            "            <<Folder>>",
            "        }",
            "    }",
            "    class Discord {",
            "        <<External Service>>",
            "    }",
        ]
    )
    styles.extend(
        [
            "    style dataset directory",
            "    style models directory",
            "    style Discord external",
        ]
    )

    for (src_mod, dst_mod), labels in sorted(relationships.items()):
        if dst_mod not in modules_found or src_mod == dst_mod:
            continue
        for label in sorted(labels):
            mermaid_lines.append(f"    {src_mod} --> {dst_mod} : {label}")

    for src_mod, folder in sorted(folder_refs):
        mermaid_lines.append(f"    {src_mod} ..> {folder} : accesses")

    if "main" in modules_found:
        mermaid_lines.append("    main --> Discord : send_notification(Webhook)")

    mermaid_lines.extend(styles)
    return "\n".join(mermaid_lines)


def write_mermaid_file(mermaid_text: str) -> None:
    MERMAID_PATH.write_text(mermaid_text + "\n", encoding="utf-8")


def generate_svg() -> None:
    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    npx_path = shutil.which("npx")
    if not npx_path:
        raise RuntimeError("npx が見つかりません。Node.js をインストールしてください。")

    subprocess.run(
        [
            npx_path,
            "-y",
            "@mermaid-js/mermaid-cli",
            "-i",
            str(MERMAID_PATH),
            "-o",
            str(SVG_PATH),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )


def build_readme_diagram_section(mermaid_text: str) -> str:
    svg_rel_path = SVG_PATH.relative_to(PROJECT_ROOT).as_posix()
    mermaid_rel_path = MERMAID_PATH.relative_to(PROJECT_ROOT).as_posix()
    return "\n".join(
        [
            README_START,
            "## システム構成図",
            "",
            "README 上では Mermaid 図をそのまま確認できます。細かい文字を拡大して見たい場合は、SVG 版を開いてブラウザのズームを使ってください。",
            "",
            f"- SVG 版: [{svg_rel_path}]({svg_rel_path})",
            f"- Mermaid ソース: [{mermaid_rel_path}]({mermaid_rel_path})",
            "",
            f"[![システム構成図 SVG]({svg_rel_path})]({svg_rel_path})",
            "",
            "```mermaid",
            mermaid_text,
            "```",
            README_END,
        ]
    )


def update_readme(diagram_section: str) -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")
    if README_START in readme_text and README_END in readme_text:
        start_index = readme_text.index(README_START)
        end_index = readme_text.index(README_END) + len(README_END)
        updated_text = (
            readme_text[:start_index] + diagram_section + readme_text[end_index:]
        )
    else:
        insertion_point = "## 📌 必要なもの"
        if insertion_point not in readme_text:
            raise RuntimeError("README に図セクションの挿入位置が見つかりません。")
        updated_text = readme_text.replace(
            insertion_point,
            diagram_section + "\n\n" + insertion_point,
            1,
        )
    README_PATH.write_text(updated_text.rstrip() + "\n", encoding="utf-8")


def generate_mermaid() -> str:
    mermaid_text = build_mermaid_text()
    write_mermaid_file(mermaid_text)
    return mermaid_text


def main() -> None:
    mermaid_text = generate_mermaid()
    # generate_svg()
    # update_readme(build_readme_diagram_section(mermaid_text))

    print("-" * 40)
    print(f"Mermaid を生成: {MERMAID_PATH}")
    print(f"SVG を生成: {SVG_PATH}")
    print(f"README を更新: {README_PATH}")
    print("-" * 40)


if __name__ == "__main__":
    main()
