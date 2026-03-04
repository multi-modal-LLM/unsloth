"""
data/pintora_docs.py

Pintora syntax documentation used as prompt context for LLM-based data generation.
Each diagram type is documented with its keywords, structure, and a short example.
"""

# ---------------------------------------------------------------------------
# Sequence Diagram
# ---------------------------------------------------------------------------

SEQUENCE_DIAGRAM_DOCS = """
## Sequence Diagram (@startsequence / @endsequence)

### Keywords
- title <text>                       — optional title
- autonumber                         — auto-number messages
- participant <name> [as <alias>]
- actor <name> [as <alias>]
- <from> -> <to>: <message>          — solid arrow (async)
- <from> --> <to>: <message>         — dashed arrow (return)
- <from> ->> <to>: <message>         — solid open arrowhead
- <from> -->> <to>: <message>        — dashed open arrowhead
- note over <participant>: <text>
- note left of <participant>: <text>
- note right of <participant>: <text>
- loop <label> ... end
- alt <condition> ... else ... end
- opt <condition> ... end
- par ... and ... end
- activate <participant> / deactivate <participant>

### Example
```
@startsequence
title User Login Flow
actor User
participant Frontend
participant AuthService
participant Database

User -> Frontend: enter credentials
Frontend ->> AuthService: POST /login
AuthService -> Database: query user
Database --> AuthService: user record
AuthService --> Frontend: JWT token
Frontend --> User: show dashboard
@endsequence
```
"""

# ---------------------------------------------------------------------------
# Component Diagram
# ---------------------------------------------------------------------------

COMPONENT_DIAGRAM_DOCS = """
## Component Diagram (@startcomponent / @endcomponent)

### Keywords
- title <text>
- component <name> [as <alias>]
- component "<name with spaces>" as <alias>
- package <name> { ... }              — group components
- database <name>
- queue <name>
- [<component>] --> [<component>]: <label>    — dependency arrow
- [<component>] ..> [<component>]: <label>    — dashed dependency

### Example
```
@startcomponent
title E-Commerce System
package "Frontend" {
  component "Web App" as WebApp
  component "Mobile App" as MobileApp
}
package "Backend" {
  component "API Gateway" as API
  component "Order Service" as Orders
  component "Auth Service" as Auth
}
database "PostgreSQL" as DB
queue "RabbitMQ" as MQ

[WebApp] --> [API]: REST
[MobileApp] --> [API]: REST
[API] --> [Auth]: validate token
[API] --> [Orders]: place order
[Orders] --> [DB]: persist
[Orders] --> [MQ]: emit event
@endcomponent
```
"""

# ---------------------------------------------------------------------------
# Entity-Relationship (ER) Diagram
# ---------------------------------------------------------------------------

ER_DIAGRAM_DOCS = """
## Entity-Relationship Diagram (@startuml erDiagram / @enduml)

### Keywords
- erDiagram (opening keyword inside @startuml block)
- ENTITY_NAME { <fields> }
  - field syntax: <type> <name>
- <ENTITY1> <cardinality> <ENTITY2>: <label>
  - cardinalities: ||--||  ||--o{  }o--||  }o--o{  ||--|{  |{--|{

### Example
```
@startuml
erDiagram
  USER {
    int id
    string name
    string email
    datetime created_at
  }
  POST {
    int id
    string title
    text body
    int user_id
  }
  COMMENT {
    int id
    text content
    int post_id
    int user_id
  }
  USER ||--o{ POST : "writes"
  POST ||--o{ COMMENT : "has"
  USER ||--o{ COMMENT : "writes"
@enduml
```
"""

# ---------------------------------------------------------------------------
# Activity Diagram
# ---------------------------------------------------------------------------

ACTIVITY_DIAGRAM_DOCS = """
## Activity Diagram (@startactivity / @endactivity)

### Keywords
- title <text>
- start / stop / end
- :<action text>;                      — activity node
- if (<condition>) then (yes/no) ... else ... endif
- while (<condition>) ... endwhile
- fork ... fork again ... end fork
- group <label> ... end group
- note: <text>

### Example
```
@startactivity
title Handle HTTP Request
start
:Receive HTTP Request;
if (authenticated?) then (yes)
  :Parse request body;
  if (valid input?) then (yes)
    :Process business logic;
    :Write to database;
    :Return 200 OK;
  else (no)
    :Return 400 Bad Request;
  endif
else (no)
  :Return 401 Unauthorized;
endif
stop
@endactivity
```
"""

# ---------------------------------------------------------------------------
# Mindmap Diagram
# ---------------------------------------------------------------------------

MINDMAP_DIAGRAM_DOCS = """
## Mindmap Diagram (@startmindmap / @endmindmap)

### Keywords
- Uses indented `*` markers to denote hierarchy depth.
- * <root>            — root node (single `*`)
- ** <child>          — level 1 child
- *** <grandchild>    — level 2 child
- Left-side branches: use `-` prefix instead of `+`
  - *- / **- for left-side nodes

### Example
```
@startmindmap
* Cloud Architecture
** Compute
*** EC2
*** Lambda
*** ECS
** Storage
*** S3
*** EBS
*** Glacier
** Networking
*** VPC
*** CloudFront
*** Route 53
** Database
*** RDS
*** DynamoDB
*** ElastiCache
@endmindmap
```
"""

# ---------------------------------------------------------------------------
# Gantt Diagram
# ---------------------------------------------------------------------------

GANTT_DIAGRAM_DOCS = """
## Gantt Diagram (@startgantt / @endgantt)

### Keywords
- title <text>
- dateFormat YYYY-MM-DD
- axisFormat <format>
- section <name>
- <task name>: <duration>         — task without ID
- <task name>: <id>, <duration>   — task with ID
- <task name>: <id>, after <id2>, <duration>   — dependency

### Example
```
@startgantt
title Sprint 1 Plan
dateFormat YYYY-MM-DD
section Design
  Write specs: a1, 2022-01-01, 3d
  Create mockups: a2, after a1, 2d
section Development
  Backend API: b1, after a1, 5d
  Frontend: b2, after a2, 4d
section Testing
  QA Testing: after b1, 3d
@endgantt
```
"""

# ---------------------------------------------------------------------------
# Class Diagram
# ---------------------------------------------------------------------------

CLASS_DIAGRAM_DOCS = """
## Class Diagram (@startclass / @endclass)

### Keywords
- title <text>
- class <Name> { <members> }
  - member prefixes: + public, - private, # protected, ~ package
  - member types: <visibility><type> <name>  or  <visibility><name>(): <returnType>
- <ClassA> --|> <ClassB>      — inheritance
- <ClassA> --* <ClassB>       — composition
- <ClassA> --o <ClassB>       — aggregation
- <ClassA> --> <ClassB>       — association
- <ClassA> ..> <ClassB>       — dependency
- <ClassA> ..|> <ClassB>      — interface implementation

### Example
```
@startclass
title Animal Kingdom
class Animal {
  +string name
  +int age
  +speak(): void
  +eat(): void
}
class Dog {
  +string breed
  +fetch(): void
}
class Cat {
  +bool indoor
  +purr(): void
}
class PoliceDog {
  +string badgeNumber
  +patrol(): void
}
Animal --|> Dog
Animal --|> Cat
Dog --|> PoliceDog
@endclass
```
"""

# ---------------------------------------------------------------------------
# Aggregated reference
# ---------------------------------------------------------------------------

DIAGRAM_TYPES = [
    "sequenceDiagram",
    "componentDiagram",
    "erDiagram",
    "activityDiagram",
    "mindmap",
    "ganttDiagram",
    "classDiagram",
]

DIAGRAM_TYPE_TO_DOCS = {
    "sequenceDiagram": SEQUENCE_DIAGRAM_DOCS,
    "componentDiagram": COMPONENT_DIAGRAM_DOCS,
    "erDiagram": ER_DIAGRAM_DOCS,
    "activityDiagram": ACTIVITY_DIAGRAM_DOCS,
    "mindmap": MINDMAP_DIAGRAM_DOCS,
    "ganttDiagram": GANTT_DIAGRAM_DOCS,
    "classDiagram": CLASS_DIAGRAM_DOCS,
}

ALL_DOCS = "\n\n".join(DIAGRAM_TYPE_TO_DOCS.values())
