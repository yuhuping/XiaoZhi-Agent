# AGENTS.md

## Project
XiaoZhi is a multimodal learning companion agent prototype for children aged 3–8.

The repository is currently in **v1.3**.

The goal of this stage is to move from a fixed workflow prototype to a more flexible **ReAct-style agent**, while adding three core agent capabilities:
- **ReAct** ReAct-style reasoning and acting loop
- **RAG** for grounded knowledge retrieval
- **MCP-oriented tools** for externalized capabilities
- **Memory** for short-term context and lightweight personalization

This is still an engineering prototype, not a production system.

---

## Current stage goal
The goal of v1.3 is to make XiaoZhi behave more like a real agent:

1. understand the child input
2. reason about what to do next
3. choose whether to respond directly, retrieve knowledge, use memory, or call a tool
4. generate a child-friendly response
5. update memory for the next turn

The priority of this stage is:

- flexible agent decision-making
- grounded responses
- reusable tools
- lightweight memory

---

## Core design shift
In v1.2, the system mainly relied on a fixed workflow.

In v1.3, the system should shift toward a **ReAct-style design**:

- **Reason**: interpret the current input and decide the next action
- **Act**: call tools, retrieve knowledge, read/write memory, or respond
- **Observe**: use tool results or retrieved context to continue generation

The workflow should remain controlled and simple, but no longer be limited to one rigid path.

---

## Product positioning
XiaoZhi is not an open-ended chatbot or emotional companion.

It is a **mode-aware child-learning agent** with two primary interaction modes:

- **Education mode**: the default learning mode, where the agent guides the child to think, answer, and learn step by step instead of simply giving answers directly.
- **Companion mode**: a lighter everyday interaction mode, where the agent responds warmly and naturally, supports simple conversation, and maintains engagement without always forcing a teaching flow.

Across both modes, XiaoZhi should:
- stay child-friendly
- remain warm, short, and clear
- avoid unsafe or manipulative interaction
- preserve a supportive educational tone
---
## Interaction modes

### Education mode
Education mode is the core mode of XiaoZhi in v1.3.

In this mode, the agent should:
- guide the child to think before answering
- ask age-appropriate follow-up questions
- provide hints step by step
- encourage partial answers and reasoning
- correct gently instead of only giving the final answer

### Companion mode
Companion mode is used for lighter everyday interaction.

In this mode, the agent should:
- respond naturally and warmly
- support short child-friendly conversation
- answer simple questions directly when appropriate
- avoid turning every turn into a teaching sequence

The agent should be able to choose or switch modes based on user input, context, and current interaction goals.
---
## Current scope

### Required
- **Backend:** FastAPI
- **Agent style:** ReAct-style reasoning and acting loop
- **Model layer:** provider-agnostic LLM integration with tool-calling capability
- **RAG:** local retrieval pipeline built from the repository's `KG/` folder
- **Tools:** LangGraph-based tool-calling layer, including prebuilt tool execution support
- **Memory:** short-term session memory and lightweight profile memory
- **Schemas:** clear request/response models
- **Local demoability:** the project must remain easy to run locally

---

## Product positioning
XiaoZhi is not an open-ended chatbot or emotional companion.

It is a **task-driven child-learning agent** that should:

- explain things simply
- ask age-appropriate follow-up questions
- answer child questions
- use retrieval when knowledge grounding is helpful
- use memory to maintain continuity
- stay warm, short, and educational

---

## ReAct workflow expectations
The agent should follow a simple ReAct-style pattern:

1. **Understand**
   - parse text/image input
   - detect topic, intent, and current teaching need
   - infer the current interaction mode when possible

2. **Reason**
   - decide the next best action
   - decide whether the agent should stay in Education mode or Companion mode

3. **Act**
   - respond directly, or
   - retrieve knowledge, or
   - read/write memory, or
   - call a tool

4. **Observe**
   - use returned context or tool output

5. **Respond**
   - generate the final child-friendly answer in the appropriate mode

6. **Update memory**
   - store useful short-term context
   - optionally update lightweight profile memory
   
The system should stay controlled and readable.
Do not turn ReAct into an overly complex autonomous loop.

---

## Code writing guidelines:
1. Prioritize completing the required task functions.
2. The code should be as concise as possible, without considering error checks at the level of actual engineering. It should be clear and straightforward, similar to pseudo-code.
3. For each function and extremely important variables, provide brief Chinese comments.
