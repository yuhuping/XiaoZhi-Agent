# AGENTS.md

## Project
XiaoZhi is a multimodal learning companion agent prototype for children aged 3–8.

The repository is currently in **v1.1**.

The goal of this stage is to build a **small, runnable, demoable prototype** that proves the core interaction loop works end to end.

This is **not** the full target architecture yet.
At this stage, the priority is:

- make the main demo path work
- keep the codebase clean and easy to extend
- preserve child-friendly behavior
- avoid overengineering

---

## Current stage goal
The goal of v1.1 is to complete one working interaction path:

1. accept image or text input
2. identify the object or topic
3. generate a short child-friendly explanation
4. ask one simple age-appropriate follow-up question
5. return the result successfully through a backend API

This stage is about building the **first stable demo**, not the full agent system.

---

## What v1.1 should demonstrate
The current stage should clearly demonstrate:

- multimodal input handling
- LLM-based topic/object understanding
- child-friendly response generation
- one complete backend request/response loop
- a code structure that can evolve into a fuller agent system later

---

## What v1.1 is not trying to do
The following are **not required yet** unless they are trivial to add:

- full workflow orchestration with LangGraph
- full MCP implementation
- tool registry or complex tool abstraction
- RAG pipeline
- short-term or long-term memory system
- parent summary generation
- streaming architecture
- structured observability / tracing system
- production deployment
- polished frontend
- multi-agent collaboration

Do not build future-stage architecture too early.

---

## Product positioning
XiaoZhi is **not** an open-ended chatbot or emotional companion.

It is a **task-driven child-learning assistant prototype**.

At this stage, the interaction style should be:

- short
- simple
- warm
- encouraging
- educational
- easy for children to understand

The prototype should focus on scenarios such as:

- recognizing a simple object from an image
- explaining it in child-friendly language
- asking one easy follow-up question
- responding to a short text-based learning prompt

---

## Current architecture scope

### Required
- **Backend:** FastAPI
- **Model layer:** Gemini for multimodal understanding and response generation
- **Prompt layer:** reusable prompt templates
- **Schemas:** clear request/response models
- **Local demoability:** the project must run locally with simple commands

### Optional but lightweight
- a simple service layer
- simple config management
- basic logging

### Not required yet
- LangGraph
- FAISS
- memory layers
- safety subsystem beyond lightweight guardrails
- SSE / WebSocket
- tool registry
- skill framework
- trace system

---

## Main v1.1 interaction path
The default flow in this stage is:

1. receive image or text input
2. detect or infer the object/topic
3. generate a short explanation suitable for ages 3–8
4. ask one simple follow-up question
5. return structured API output

This path should remain stable throughout v1.1.

If a change makes this path less reliable, it is probably not suitable for the current stage.

---

## Minimal tools mindset
v1.1 does not need a formal tool system, but the code should already reflect a **tool-oriented mindset**.

Simple helper functions or service functions may act as early-stage tools, for example:

- `detect_topic`
- `generate_explanation`
- `generate_followup_question`

These do not need a registry or complex abstraction yet.
They just need clear responsibilities.

---

## Minimal skills mindset
v1.1 does not need a formal skills layer, but the project should already revolve around one core task-level capability:

### ExplainAndAskSkill
This implicit skill should do the following:

- understand the child input
- identify the topic or object
- explain it in child-friendly language
- ask one simple follow-up question

This skill does not need its own framework yet, but the design should keep this capability clear.

---

## Implementation principles for v1.1

### 1. Keep the first path small
Build one strong happy path first.
Do not branch into many features.

### 2. Prefer readability over abstraction
At this stage, simple and clear code is better than flexible but premature architecture.

### 3. Keep route handlers thin
API routes should mainly:
- validate input
- call service logic
- return structured output

Model calls and prompt logic should live outside route handlers.

### 4. Keep prompts reusable
Prompt text should not be scattered across many files.
Store reusable prompt templates centrally.

### 5. Preserve future extensibility
The code does not need full workflow/tool/skill architecture yet, but it should not block that evolution later.

### 6. Keep outputs child-appropriate
Responses should remain short, supportive, and easy to understand.

---

## Suggested repository structure for v1.1
The codebase should stay close to the following structure:

- `app/api/`
  - FastAPI endpoints
  - request/response handling

- `app/schemas/`
  - pydantic models for input/output

- `app/services/`
  - multimodal input handling
  - model invocation
  - response generation logic

- `app/prompts/`
  - child tutoring prompts
  - explanation prompts
  - follow-up question prompts

- `app/core/`
  - config
  - environment setup
  - shared utilities

- `tests/`
  - basic endpoint tests
  - minimal service tests

- `scripts/`
  - local run helpers
  - debug helpers

Do not introduce many more top-level modules unless clearly necessary.

---

## Suggested minimal modules
The first implementation will likely need modules equivalent to:

- `app/main.py`
- `app/api/chat.py`
- `app/schemas/chat.py`
- `app/services/chat_service.py`
- `app/services/model_service.py`
- `app/prompts/tutor_prompts.py`
- `app/core/config.py`

The exact names may vary, but responsibilities should stay simple and explicit.

---

## Prompt behavior requirements
Prompts in v1.1 should guide the model to:

- speak in simple child-friendly language
- keep answers short
- explain clearly
- stay warm and encouraging
- ask one follow-up question
- avoid long open-ended chatting
- avoid sounding like a parent, therapist, or best friend

Do not build a complicated prompt system yet.
A few clean reusable templates are enough.

---

## Minimal safety expectations
A full safety subsystem is not required yet, but the current prototype must still follow basic child-facing constraints.

### Must do
- keep responses age-appropriate
- avoid harmful or unsafe suggestions
- avoid adult or explicit content
- avoid manipulative or emotionally dependent phrasing
- stay focused on simple educational interaction

### Must not do
- do not claim to be a real person or guardian
- do not encourage dangerous behavior
- do not generate disturbing or inappropriate content
- do not turn the interaction into unrestricted chatting

If needed, lightweight safety constraints may be implemented through prompts or simple post-processing.

---

## API expectations
At minimum, the backend should expose:

- one working endpoint for the main interaction path
- one health check endpoint

The response should preferably be structured rather than raw free-form text.

A useful response shape may include:

- detected topic or object
- explanation text
- follow-up question
- optional metadata

Exact field names may change, but structured output is preferred.

---

## Logging expectations
Heavy observability is not needed yet.
However, basic logs are encouraged for:

- request received
- model call started
- model call completed
- major failure cases

Keep logging lightweight and useful.

---

## Testing expectations
Testing in v1.1 should stay lightweight.

Prefer:

- one endpoint smoke test
- one or two service-level sanity tests
- one invalid-input test

Do not spend too much time building heavy test infrastructure in this stage.

---

## Definition of done for v1.1
A task in the current stage is done only if:

- the code runs locally
- the main demo endpoint still works
- image or text input produces a valid child-friendly response
- the response includes an explanation and one follow-up question
- changed code remains easy to read
- the structure still supports later extension

---

## Current non-goals
The following are explicitly out of scope in v1.1:

- full workflow graph
- retrieval engine
- memory system
- parent summary
- streaming output
- trace system
- complex frontend
- deployment optimization
- evaluation framework
- external MCP server implementation

---

## Commands
Fill these in as the repo evolves.

- install: `pip install -r requirements.txt`
- run backend: from repo root use `python -m uvicorn app.main:app --reload`; from `app/` use `python -m uvicorn main:app --reload`
- run tests: `pytest`
- lint: not added yet
- format: not added yet

Keep these commands stable once they are introduced.

---

## Collaboration instructions for coding agents
When working in this repository during v1.1:

1. first understand the current demo path
2. protect the runnable end-to-end flow
3. prefer small changes over large refactors
4. do not introduce later-stage architecture unless necessary
5. keep route handlers thin
6. keep prompts reusable and centralized
7. keep code easy for a human developer to continue
8. when uncertain, choose the simpler implementation

For this stage, optimize for:

- runnable demo
- clean structure
- child-friendly output
- future extensibility without overengineering

---

## Repo intent in the current stage
At v1.1, this repository should already show a real prototype foundation:

- multimodal input
- child-friendly explanation
- one follow-up teaching question
- clean backend structure
- clear path toward a fuller agent system later

The current stage should be easy to run, easy to demo, and easy to extend.
