"""
Imports: inspect, importlib, traceback
Inputs: tool_package (str), base_tool_path (str)
Outputs: dict - Tool validation report.
Exceptions: ImportError, AttributeError
Dependencies: Standard library
"""
import inspect
import importlib
import traceback

def validate_tools(tool_package="app.tools", base_class_path="app.core.base_tool.BaseTool"):
    results = {"passed": True, "valid_tools": [], "invalid_tools": [], "errors": []}
    
    try:
        base_module_name, base_class_name = base_class_path.rsplit(".", 1)
        base_module = importlib.import_module(base_module_name)
        BaseTool = getattr(base_module, base_class_name)
    except Exception:
        results["passed"] = False
        results["errors"].append({"trace": traceback.format_exc(), "fix": f"Ensure BaseTool exists at {base_class_path}"})
        return results

    try:
        tools_module = importlib.import_module(tool_package)
        for name, obj in inspect.getmembers(tools_module, inspect.isclass):
            if obj is not BaseTool and issubclass(obj, BaseTool):
                if not hasattr(obj, "run") or not callable(getattr(obj, "run")):
                    results["passed"] = False
                    results["invalid_tools"].append(name)
                    results["errors"].append({
                        "tool": name,
                        "fix": f"Tool {name} is missing a callable 'run' method."
                    })
                else:
                    results["valid_tools"].append(name)
    except ImportError:
        results["errors"].append({"trace": traceback.format_exc(), "fix": f"Check tool package path {tool_package}."})

    return results

if __name__ == "__main__":
    import json
    print(json.dumps(validate_tools(), indent=2))