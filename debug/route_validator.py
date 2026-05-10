"""
Imports: importlib, traceback, fastapi.FastAPI
Inputs: app_path (str) - Import path to the FastAPI app instance.
Outputs: dict - Status of routes, missing typed responses.
Exceptions: ImportError, AttributeError
Dependencies: fastapi
"""
import importlib
import traceback

def validate_routes(app_path="app.main.app"):
    results = {"passed": True, "routes": [], "warnings": [], "errors": []}
    
    try:
        module_name, app_name = app_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        app = getattr(module, app_name)
        
        from fastapi import routing
        for route in app.routes:
            if isinstance(route, routing.APIRoute):
                results["routes"].append(f"{route.methods} {route.path}")
                if not route.response_model and not route.response_class:
                    results["warnings"].append(f"Route {route.path} lacks a defined response_model.")
    except Exception as e:
        results["passed"] = False
        results["errors"].append({"trace": traceback.format_exc(), "fix": f"Ensure FastAPI app exists at {app_path}"})

    return results

if __name__ == "__main__":
    import json
    print(json.dumps(validate_routes(), indent=2))