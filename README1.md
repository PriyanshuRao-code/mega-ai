# mega-ai
================================================
FILES TO CREATE
================================================
api/
|-- routes.py
|-- sse.py
|-- middleware.py
|-- error_handlers.py
================================================
ENDPOINTS REQUIRED
================================================
1. submit query
2. execution trace
3. latest eval summary
4. approve/reject rewrite
5. targeted reevaluation
================================================
INPUT / OUTPUT
================================================
All endpoints must:
- use pydantic models
- return structured responses

FileResponsibility
api/models.py
api/routes.py
api/sse.py 
api/middleware.py
api/error_handlers.py
api/app.py
debug/run_api_debug.py