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

def validate_agents(agent_package="app.agents", base_class_path="app.core.base_agent.BaseAgent"):
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
        agents_module = importlib.import_module(agent_package)
        for name, obj in inspect.getmembers(agents_module, inspect.isclass):
            if obj is not BaseAgent and issubclass(obj, BaseAgent):
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