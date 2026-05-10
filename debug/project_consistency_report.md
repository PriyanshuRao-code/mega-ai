# Project Consistency & Integration Report

## 1. Executive Summary

This report outlines the architectural consistency, integration compatibility, and systemic risks within the multi-agent system. The validation was performed using the `debug/full_system_check.py` tool. While the core interfaces and components are present, several major schisms exist between modules that were generated independently, particularly concerning data validation frameworks and duplicated contracts.

## 2. Architecture Mismatches

### The Pydantic vs. Dataclasses Schism
The most significant architectural mismatch is the fragmented approach to data validation and serialization. The project is split down the middle:

**Using Pydantic:**
- `api/error_handlers.py`
- `api/models.py`
- `contracts/eval_contracts.py`
- `contracts/event_contracts.py`

**Using Dataclasses:**
- `contracts/agent_contracts.py`
- `contracts/models.py`
- `contracts/shared_context_prompt2.py`
- `contracts/shared_context_prompt3.py`
- `contracts/tool_contracts.py`
- `evals/datasets.py`
- `interfaces/base_agent.py`
- `orchestrator/retry_manager.py`
- `orchestrator/router.py`

**Risk:** High. `SharedContext` and other internal models defined as `dataclass`es will fail when passed into API models or Event contracts expecting Pydantic `BaseModel`s, leading to runtime serialization/validation errors.

## 3. Duplicate Contracts & Import Inconsistencies

Several core contracts have been duplicated during independent generation, leading to fragmented imports across the system:

- **`SharedContext`**:
  - `contracts/models.py`
  - `contracts/shared_context_prompt2.py`
  - `contracts/shared_context_prompt3.py`
  - *Risk*: Agents relying on different definitions of `SharedContext` will fail type checks and potentially crash if attributes differ.
  
- **`BaseAgent`**:
  - `interfaces/base.py`
  - `interfaces/base_agent.py`
  - *Risk*: Agents subclassing different `BaseAgent` definitions will break orchestrator type-checks or missing methods required by the router.

## 4. Inconsistent Inheritance

- **Resolved**: All agents (`CompressionAgent`, `CritiqueAgent`, `DecompositionAgent`, `RetrievalAgent`, `SynthesisAgent`) correctly inherit from the `BaseAgent` Generic (`BaseAgent[SharedContext, ...]`).
- **Resolved**: All tools correctly inherit from `BaseTool`.

*(Note: Initial AST parsing flagged these as errors because they inherited from `BaseAgent[SharedContext, Result]`, which was a subscript rather than a direct name. The validator script was updated to correctly identify Generic inheritance).*

## 5. Dangerous Assumptions

- **Environment Configuration**: The system heavily relies on `.env` configuration (`POSTGRES_DB`, `API_SECRET_KEY`). While `.env.example` exists and is well-documented, independent modules may lack robust fallbacks or runtime assertions if these variables are omitted, leading to silent failures or generic tracebacks rather than explicit configuration errors.
- **Dependency Assumptions**: The worker service assumes `api` and `postgres` are healthy before starting, but if `api` crashes due to the Pydantic/Dataclass schism, the worker will fail to boot or constantly restart.

## 6. Unresolved TODOs

- Codebase analysis indicates no explicit `TODO` markers. However, resolving the duplicate contracts and the Pydantic/Dataclass schism serves as the primary implicit TODO for the integration phase.

## 7. Integration Risk Assessment

| Component | Risk Level | Description | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Data Models** | **CRITICAL** | API expects Pydantic, Orchestrator/Agents expect Dataclasses. | Standardize on **Pydantic** system-wide, given its advantages for API boundaries and runtime validation. |
| **Contracts** | **HIGH** | Multiple source-of-truth files for `SharedContext` and `BaseAgent`. | Delete `prompt2`/`prompt3` legacy files. Consolidate into `contracts/models.py` and `interfaces/base_agent.py`. |
| **Orchestrator** | **MEDIUM** | Requires unified `BaseAgent` to route correctly. | Ensure all agents import `BaseAgent` from `interfaces.base_agent`, not `interfaces.base`. |
| **API/Worker** | **LOW** | Docker compose dependencies correctly implemented. | Fix data models before full integration testing. |

## 8. Next Steps for Integration

1. **Refactor `contracts/`**: Remove `shared_context_prompt2.py` and `shared_context_prompt3.py`. Consolidate all shared state models into `contracts/models.py` using Pydantic.
2. **Refactor `interfaces/`**: Remove `interfaces/base.py`. Ensure all agents use `interfaces/base_agent.py`.
3. **Unify Validation**: Convert `dataclass` models in `agent_contracts.py` and `tool_contracts.py` to Pydantic `BaseModel`s.
4. **Integration Test Run**: Re-run `debug/full_system_check.py` to ensure 0 duplicates and 0 dataclasses remain in the contract layer.
