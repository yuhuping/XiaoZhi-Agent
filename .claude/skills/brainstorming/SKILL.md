---
name: brainstorming
description: Use when starting any new feature, change, or improvement for XiaoZ before writing code - explores intent, requirements, and design before implementation. Use when asked to add functionality, fix behavior, or build new components.
---

# Brainstorming for XiaoZ

Turn ideas into fully formed designs through collaborative dialogue before touching any code.

<HARD-GATE>
Do NOT write any code, invoke any implementation action, or scaffold anything until you have presented a design and the user has explicitly approved it.
</HARD-GATE>

## XiaoZ Project Context

小智 is a ReAct-based educational agent with three modes: `education` (guided learning), `companion` (casual chat), `parent` (practical advice + web search). Built on LangGraph with RAG, four-tier memory, and SSE streaming.

Key architectural constraints to keep in mind:
- LangGraph DAG is fixed: `understand → state_update → chatbot → tools → observe → respond → memory_update → response → memory_compact → END`
- Settings are frozen at import time — server restart needed after `.env` changes
- Three act routes in `router.py`: `direct`, `retrieve_knowledge`, `read_memory`
- Frontend is a single `app/frontend/index.html` — no build step

## Checklist

Create a task for each item and complete in order:

1. **Explore project context** — read relevant files, schemas, nodes, prompts
2. **Ask clarifying questions** — one at a time, understand purpose/constraints/success criteria
3. **Propose 2-3 approaches** — with trade-offs and your recommendation
4. **Present design** — section by section, get approval after each
5. **Write design doc** — save to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
6. **Spec self-review** — fix placeholders, contradictions, ambiguity inline
7. **User reviews spec** — wait for approval before proceeding
8. **Transition** — invoke `superpowers:writing-plans` to create implementation plan

## Process

**Before asking questions:**
- Read the relevant code (schemas, nodes, prompts, frontend) to understand current state
- Identify which part of the pipeline the change touches
- Note any existing patterns to follow (e.g., how other nodes are structured)

**Clarifying questions (one at a time):**
- What mode(s) does this affect? (education / companion / parent / all)
- Is this a frontend change, backend change, or both?
- Does it affect the LangGraph DAG, prompts, memory, RAG, or API schema?
- What is the success criteria?

**Propose 2-3 approaches** with trade-offs. Lead with your recommendation.

**Design sections to cover:**
- Which files change and why
- Data flow through the pipeline
- Prompt / schema changes (if any)
- Frontend impact (if any)
- How to test it

## Anti-Pattern: "Too Simple to Design"

Even small changes — adding an option to the frontend, tweaking a prompt, adding a field to the schema — get this process. "Simple" changes in a pipeline like XiaoZ often have unexpected downstream effects (state propagation, prompt rendering, SSE output). The design can be two sentences, but you MUST present it.

## After Design Approval

Write spec to `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`, self-review, then:

> "Spec written to `<path>`. Please review it and let me know if you want any changes before we write the implementation plan."

Only invoke `superpowers:writing-plans` after user approves the spec.

## Key Principles

- One question at a time
- Multiple choice preferred over open-ended
- YAGNI — remove unnecessary features from all designs
- Follow existing patterns in the codebase
- Don't propose unrelated refactoring
