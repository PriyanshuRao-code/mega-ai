import os
import ast
import json
import sys

def get_python_files(root_dir):
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if 'venv' in dirpath or '__pycache__' in dirpath or 'TEMP' in dirpath or 'PROMPTS' in dirpath:
            continue
        for f in filenames:
            if f.endswith('.py'):
                py_files.append(os.path.join(dirpath, f))
    return py_files

def analyze_ast(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError as e:
            return {"error": str(e)}

    imports = []
    classes = []
    bases = []
    decorators = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append((node.name, b.id))
                elif isinstance(b, ast.Attribute):
                    bases.append((node.name, b.attr))
                elif isinstance(b, ast.Subscript):
                    if isinstance(b.value, ast.Name):
                        bases.append((node.name, b.value.id))
                    elif isinstance(b.value, ast.Attribute):
                        bases.append((node.name, b.value.attr))
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.append((node.name, dec.id))
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    decorators.append((node.name, dec.func.id))
                    
    return {
        "imports": imports,
        "classes": classes,
        "bases": bases,
        "decorators": decorators
    }

def check_env_assumptions(root_dir):
    assumptions = []
    env_file = os.path.join(root_dir, '.env.example')
    if not os.path.exists(env_file):
        assumptions.append("Missing .env.example file")
    else:
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'POSTGRES_DB' not in content:
                assumptions.append("Missing POSTGRES_DB in .env.example")
    return assumptions

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    files = get_python_files(root)
    
    report = {
        "duplicate_contracts": {},
        "pydantic_dataclass_mix": {"pydantic": [], "dataclasses": []},
        "inheritance_issues": [],
        "broken_interfaces": [],
        "api_schema_issues": [],
        "docker_env_assumptions": check_env_assumptions(root),
        "import_issues": []
    }
    
    contract_definitions = {}
    
    for py_file in files:
        rel_path = os.path.relpath(py_file, root).replace('\\', '/')
        if rel_path.startswith('debug/'):
            continue
            
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'pydantic' in content:
                report["pydantic_dataclass_mix"]["pydantic"].append(rel_path)
            if 'dataclasses' in content:
                report["pydantic_dataclass_mix"]["dataclasses"].append(rel_path)
        
        info = analyze_ast(py_file)
        if "error" in info:
            continue
            
        for cls in info["classes"]:
            if cls not in contract_definitions:
                contract_definitions[cls] = []
            contract_definitions[cls].append(rel_path)
            
        if rel_path.startswith('agents/'):
            agent_classes = [c for c in info["classes"] if 'Agent' in c]
            for cls, base in info["bases"]:
                if cls in agent_classes and base != 'BaseAgent' and base != 'ABC':
                    report["inheritance_issues"].append(f"{cls} in {rel_path} inherits from {base} instead of BaseAgent")
            for cls in agent_classes:
                has_base = any(b == 'BaseAgent' for c, b in info["bases"] if c == cls)
                if not has_base:
                    report["inheritance_issues"].append(f"{cls} in {rel_path} does not inherit from BaseAgent")
                    
        if rel_path.startswith('tools/'):
            tool_classes = [c for c in info["classes"] if 'Tool' in c]
            for cls, base in info["bases"]:
                if cls in tool_classes and base != 'BaseTool' and base != 'ABC':
                    report["inheritance_issues"].append(f"{cls} in {rel_path} inherits from {base} instead of BaseTool")
            for cls in tool_classes:
                has_base = any(b == 'BaseTool' for c, b in info["bases"] if c == cls)
                if not has_base:
                    report["inheritance_issues"].append(f"{cls} in {rel_path} does not inherit from BaseTool")

    for cls, paths in contract_definitions.items():
        if len(paths) > 1 and cls in ['SharedContext', 'BaseAgent', 'BaseTool', 'AgentTask', 'Event', 'ModelConfig', 'RetryConfig', 'LogEvent', 'AgentState']:
            report["duplicate_contracts"][cls] = paths

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()