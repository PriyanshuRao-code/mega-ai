"""
Imports: importlib, traceback, fastapi.FastAPI
Inputs: app_path (str) - Import path to the FastAPI app instance.
Outputs: dict - Status of routes, missing typed responses.
Exceptions: ImportError, AttributeError
Dependencies: fastapi
"""
import importlib
import traceback

import os
import sys

# Inject repo root into sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_routes(app_path="api.app.create_app"):
    results = {"passed": True, "routes": [], "warnings": [], "errors": []}
    
    try:
        module_name, app_name = app_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        app_obj = getattr(module, app_name)
        app = app_obj() if callable(app_obj) else app_obj
        
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