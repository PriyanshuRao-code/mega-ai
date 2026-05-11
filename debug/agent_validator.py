"""
Imports: inspect, importlib, traceback
Inputs: agent_module_path (str), base_agent_path (str)
Outputs: dict - Pass/fail status, invalid agents, missing methods.
Exceptions: ImportError, AttributeError
Dependencies: Standard library (assumes core system has BaseAgent)
"""
import inspect
import importlib
import traceback

import os
import sys

# Inject repo root into sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_agents(agent_package="agents", base_class_path="interfaces.base_agent.BaseAgent"):
    results = {"passed": True, "valid_agents": [], "invalid_agents": [], "errors": []}
    
    try:
        base_module_name, base_class_name = base_class_path.rsplit(".", 1)
        base_module = importlib.import_module(base_module_name)
        BaseAgent = getattr(base_module, base_class_name)
    except Exception as e:
        results["passed"] = False
        results["errors"].append({"trace": traceback.format_exc(), "fix": f"Ensure BaseAgent exists at {base_class_path}"})
        return results

    try:
        import pkgutil
        
        agents_module = importlib.import_module(agent_package)
        _file_attr = getattr(agents_module, '__file__', None)
        agent_dir = os.path.dirname(_file_attr) if _file_attr else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), agent_package.replace('.', '/'))

        # Discover all submodules in the package
        modules_to_inspect = [agents_module]
        if os.path.isdir(agent_dir):
            for filename in os.listdir(agent_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = f"{agent_package}.{filename[:-3]}"
                    try:
                        modules_to_inspect.append(importlib.import_module(module_name))
                    except Exception as e:
                        results["errors"].append({"trace": traceback.format_exc(), "fix": f"Failed to import agent module {module_name}."})

        # Check all discovered modules
        seen_classes = set()
        for mod in modules_to_inspect:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if obj is not BaseAgent and issubclass(obj, BaseAgent) and obj not in seen_classes:
                    seen_classes.add(obj)
                    # Check if abstract methods are implemented
                    if inspect.isabstract(obj):
                        results["passed"] = False
                        results["invalid_agents"].append(name)
                        results["errors"].append({
                            "agent": name,
                            "fix": f"Agent {name} fails to implement required abstract methods from BaseAgent."
                        })
                    else:
                        results["valid_agents"].append(name)
    except ImportError as e:
        results["errors"].append({"trace": traceback.format_exc(), "fix": f"Check agent package path {agent_package}."})

    return results

if __name__ == "__main__":
    import json
    print(json.dumps(validate_agents(), indent=2))