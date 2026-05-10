"""
Imports: inspect, importlib, pydantic.BaseModel, traceback
Inputs: module_names (list) - List of modules containing schemas.
Outputs: dict - Schema validation report including mismatches and errors.
Exceptions: Catches TypeError, pydantic.ValidationError
Dependencies: pydantic
"""
import inspect
import importlib
import traceback
from pydantic import BaseModel

def validate_schemas(schema_modules=["app.schemas"]):
    results = {"passed": True, "validated": [], "mismatches": [], "errors": []}
    
    for mod_name in schema_modules:
        try:
            module = importlib.import_module(mod_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModel) and obj is not BaseModel:
                    try:
                        # Attempt to resolve forward refs
                        obj.model_rebuild()
                        results["validated"].append(name)
                    except Exception as e:
                        results["passed"] = False
                        results["mismatches"].append(name)
                        results["errors"].append({
                            "schema": name,
                            "trace": traceback.format_exc(),
                            "fix": f"Check field definitions and typing imports for schema {name}."
                        })
        except ImportError:
            results["errors"].append({"trace": f"Could not import {mod_name}", "fix": f"Ensure {mod_name} exists."})
            
    return results

if __name__ == "__main__":
    import json
    print(json.dumps(validate_schemas(), indent=2))