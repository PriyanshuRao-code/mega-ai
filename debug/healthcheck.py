"""
Imports: importlib, traceback
Inputs: None
Outputs: dict - Aggregated results of all individual validators.
Exceptions: ImportError
Dependencies: All sibling validator files
"""
import traceback

# List of validator modules and their execution functions
VALIDATORS = {
    "imports": ("import_validator", "validate_imports"),
    "schemas": ("schema_validator", "validate_schemas"),
    "agents": ("agent_validator", "validate_agents"),
    "tools": ("tool_validator", "validate_tools"),
    "routes": ("route_validator", "validate_routes"),
    "dependencies": ("dependency_validator", "validate_dependencies"),
    "integration": ("integration_test", "run")
}

def run_healthchecks():
    full_report = {}
    for key, (mod_name, func_name) in VALIDATORS.items():
        try:
            mod = __import__(mod_name)
            func = getattr(mod, func_name)
            full_report[key] = func()
        except Exception as e:
            full_report[key] = {
                "passed": False, 
                "errors": [{"trace": traceback.format_exc(), "fix": f"Fix validator {mod_name}.py"}]
            }
    return full_report

if __name__ == "__main__":
    import json
    print(json.dumps(run_healthchecks(), indent=2))