"""
Imports: os, sys, importlib, traceback, pathlib
Inputs: target_directory (str) - Default is project root.
Outputs: dict - Contains 'passed' (bool), 'missing_imports' (list), 'circular_errors' (list), 'suggestions' (list)
Exceptions: Handles ImportError, ModuleNotFoundError
Dependencies: Standard Python library
"""
import os
import sys
import importlib
import traceback
from pathlib import Path

def validate_imports(target_dir="."):
    print("========================================")
    print(" IMPORT VALIDATOR")
    print("========================================")
    results = {
        "passed": True,
        "missing_imports": [],
        "circular_errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    base_path = Path(target_dir).resolve()
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))

    # Only scan core multi-agent system directories
    allowed_dirs = {'contracts', 'interfaces', 'orchestrator', 'agents', 'tools', 'api', 'infrastructure'}
    
    python_files = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        if os.path.abspath(dirpath) == os.path.abspath(base_path):
            dirnames[:] = [d for d in dirnames if d in allowed_dirs]
        else:
            dirnames[:] = [d for d in dirnames if not (d.startswith('.') or d.startswith('$') or d in ('venv', '__pycache__', 'TEMP', 'PROMPTS'))]
            
        for f in filenames:
            if f.endswith('.py') and not f.startswith('__'):
                python_files.append(Path(dirpath) / f)
                
    for path in python_files:
        module_path = path.relative_to(base_path).with_suffix('').parts
        module_name = ".".join(module_path)
        
        print(f"Validating import: {module_name} ... ", end="", flush=True)
        
        try:
            importlib.import_module(module_name)
            print("OK")
        except ModuleNotFoundError as e:
            print("FAILED (ModuleNotFoundError)")
            results["passed"] = False
            results["missing_imports"].append(f"{module_name}: {str(e)}")
            results["suggestions"].append(f"Fix: Install missing dependency or check path for {module_name}. Error: {e}")
        except ImportError as e:
            print("FAILED (ImportError)")
            results["passed"] = False
            error_trace = traceback.format_exc()
            if "most likely due to a circular import" in str(e).lower() or "cannot import name" in str(e):
                results["circular_errors"].append(f"{module_name}: {str(e)}\n{error_trace}")
                results["suggestions"].append(f"Fix: Refactor {module_name} to remove circular dependency.")
            else:
                results["missing_imports"].append(f"{module_name}: {str(e)}\n{error_trace}")
                
    return results

if __name__ == "__main__":
    import json
    print(json.dumps(validate_imports(".."), indent=2))