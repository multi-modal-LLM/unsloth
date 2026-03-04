"""
eval/prompts.py

Entity / action / diagram-type pools used to generate randomized evaluation prompts.
Matches the lists from the blog post exactly, plus extras for broader coverage.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pools (from the blog post)
# ---------------------------------------------------------------------------

ENTITIES = [
    "User",
    "Client",
    "WebApp",
    "Backend",
    "Server",
    "Database",
    "AuthService",
    "PaymentGateway",
    "Cache",
    "Redis",
    "Worker",
    "TaskQueue",
    "Frontend",
    "API Gateway",
    "OrderSystem",
    "Inventory",
    "NotificationSvc",
    "Logger",
    "MetricsSvc",
    # extras
    "CDN",
    "LoadBalancer",
    "EmailService",
    "SearchService",
    "FileStorage",
    "BatchProcessor",
    "ReportingService",
    "AdminPanel",
]

ACTIONS = [
    "requests login",
    "fetches data",
    "updates record",
    "processes payment",
    "validates token",
    "sends email",
    "renders view",
    "queries index",
    "health check",
    "ack signal",
    "authenticates user",
    "writes to log",
    "queries for user profile",
    "returns 200 OK",
    "returns 404 Not Found",
    "submits form",
    "enqueues job",
    "dequeues job",
    "generates report",
    # extras
    "uploads file",
    "downloads file",
    "invalidates cache",
    "retries request",
    "notifies subscriber",
    "syncs data",
]

# Diagram types supported in the evaluation (subset used in the blog post)
EVAL_DIAGRAM_TYPES = [
    "sequenceDiagram",
    "componentDiagram",
    "activityDiagram",
]

# All diagram types (for more comprehensive evaluation)
ALL_DIAGRAM_TYPES = [
    "sequenceDiagram",
    "componentDiagram",
    "activityDiagram",
    "erDiagram",
    "mindmap",
    "ganttDiagram",
    "classDiagram",
]
