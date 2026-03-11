import ast
from pathlib import Path


def generate_mermaid():
    target_dirs = ["src", "tools"]
    mermaid_lines = [
        "classDiagram",
        "    %% スタイルの定義",
        "    classDef core fill:#f96,stroke:#333,stroke-width:2px;  %% 推論（中心）",
        "    classDef tool fill:#dfd,stroke:#333;                 %% ツール系",
        "    classDef main_logic fill:#cef,stroke:#333;           %% メイン実行系",
        "    classDef directory fill:#eee,stroke:#666,stroke-dasharray: 5 5; %% フォルダ系"
    ]

    styles = []
    modules_found = []

    # 依存関係を管理
    relationships = {}
    # フォルダ参照を管理 (モジュール名, 参照先フォルダ)
    folder_refs = set()

    for d in target_dirs:
        dir_path = Path(d)
        if not dir_path.exists(): continue

        mermaid_lines.append(f"    namespace {d} {{")

        for py_file in dir_path.glob("*.py"):
            if py_file.name == "__init__.py": continue

            module_name = py_file.stem
            modules_found.append(module_name)

            mermaid_lines.append(f"        class {module_name} {{")
            mermaid_lines.append("            <<Module>>")

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                # --- ① import マッピング ---
                import_map = {}
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom) and node.module:
                        target_mod = node.module.split('.')[-1]
                        for n in node.names:
                            import_map[n.name] = target_mod
                    elif isinstance(node, ast.Import):
                        for n in node.names:
                            target_mod = n.name.split('.')[-1]
                            import_map[n.name] = target_mod

                # --- ② 定義（関数・定数）の抽出 ---
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        args = [arg.arg for arg in node.args.args]
                        mermaid_lines.append(f"            +{node.name}({', '.join(args)})")
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                mermaid_lines.append(f"            +{target.id}")

                # --- ③ 関数呼び出し ＆ パス参照 の抽出 ---
                for node in ast.walk(tree):
                    # フォルダパス（文字列）の参照を検知
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        val = node.value.lower()
                        if "dataset" in val:
                            folder_refs.add((module_name, "dataset"))
                        if "models/" in val or ".pt" in val or ".onnx" in val:
                            folder_refs.add((module_name, "models"))

                    # 関数呼び出しを検知
                    if isinstance(node, ast.Call):
                        func_name = ""
                        target_mod = ""

                        if isinstance(node.func, ast.Name):
                            func_name = node.func.id
                            if func_name in import_map:
                                target_mod = import_map[func_name]
                        elif isinstance(node.func, ast.Attribute):
                            func_name = node.func.attr
                            if isinstance(node.func.value, ast.Name) and node.func.value.id in import_map:
                                target_mod = import_map[node.func.value.id]

                        if target_mod:
                            args_str = "..."
                            if hasattr(ast, "unparse"):
                                args_list = [ast.unparse(arg).replace('"', "'") for arg in node.args]
                                args_str = ", ".join(args_list)
                                if len(args_str) > 30: args_str = args_str[:27] + "..."

                            label = f"{func_name}({args_str})"
                            rel_key = (module_name, target_mod)
                            if rel_key not in relationships:
                                relationships[rel_key] = set()
                            relationships[rel_key].add(label)

            except Exception as e:
                print(f"❌ {py_file} の解析中にエラー: {e}")

            mermaid_lines.append("        }")

            # スタイル指定
            if module_name == "inference": styles.append(f"    style {module_name} core")
            elif d == "tools": styles.append(f"    style {module_name} tool")
            elif module_name == "main": styles.append(f"    style {module_name} main_logic")

        mermaid_lines.append("    }")

    # --- ④ データ・モデルフォルダの定義を追加 ---
    mermaid_lines.append("    namespace directories {")
    mermaid_lines.append("        class dataset {")
    mermaid_lines.append("            <<Folder>>")
    mermaid_lines.append("        }")
    mermaid_lines.append("        class models {")
    mermaid_lines.append("            <<Folder>>")
    mermaid_lines.append("        }")
    mermaid_lines.append("    }")
    styles.append("    style dataset directory")
    styles.append("    style models directory")

    # --- ⑤ 依存関係（モジュール間）の書き出し ---
    for (src_mod, dst_mod), labels in relationships.items():
        if dst_mod in modules_found and src_mod != dst_mod:
            for label in labels:
                mermaid_lines.append(f"    {src_mod} --> {dst_mod} : {label}")

    # --- ⑥ フォルダパス参照の書き出し（点線） ---
    for src_mod, folder in folder_refs:
        mermaid_lines.append(f"    {src_mod} ..> {folder} : accesses")

    mermaid_lines.extend(styles)

    # 5. ファイル書き出し
    output_file = "custom_diagram.mmd"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(mermaid_lines))

    print("-" * 30)
    print(f"✅ フォルダ参照を含む詳細グラフを '{output_file}' に生成しました。")
    print("-" * 30)

if __name__ == "__main__":
    generate_mermaid()
