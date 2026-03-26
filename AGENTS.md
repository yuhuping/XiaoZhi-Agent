# AGENTS.md

## Project
XiaoZhi is a multimodal learning companion agent prototype for children aged 3–8.

The repository is currently in **v1.2**.

The goal of this stage is to upgrade the v1.1 runnable prototype into a **workflow-based agent core** with explicit state, explicit node boundaries, and a clearer path toward a full agent system.

This is still **not** the final architecture.
At this stage, the priority is:

- move from linear request handling to explicit workflow execution
- introduce a visible agent state
- split core logic into nodes with clear responsibilities
- preserve the existing runnable demo path
- prepare for later integration of retrieval, memory, safety, and summary

---

## Current stage goal
The goal of v1.2 is to transform the prototype from:

- a single-path multimodal demo

into:

- a small but real **workflow-driven agent**

The current stage should support one stable interaction path with explicit steps:

1. receive image or text input
2. perceive or infer the topic/object
3. update current agent state
4. plan the next action
5. generate a child-friendly response
6. return the result through the backend API

The system should now make the workflow visible in code.

---

## What v1.2 should demonstrate
The current stage should clearly demonstrate:

- explicit workflow orchestration
- explicit state management
- node-based separation of responsibilities
- multimodal input handling
- child-friendly response generation
- a code structure that is meaningfully closer to a real agent system

This stage should make it obvious that the project is no longer just “an API calling a model.”


---

## Current architecture scope

### Required
- **Backend:** FastAPI
- **Workflow orchestration:** LangGraph or an equivalent explicit graph-based workflow layer
- **State layer:** a typed and inspectable agent state
- **Model layer:** OpenAI for multimodal understanding and response generation
- **Prompt layer:** reusable prompt templates
- **Schemas:** clear request/response models
- **Local demoability:** the project must remain easy to run locally

