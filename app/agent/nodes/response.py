from __future__ import annotations

import logging

from app.agent.state import AgentState, append_trace
from app.schemas.chat import ChatMetadata, ChatResponse, GroundingInfo, GroundingSource, MemoryInfo, ReactInfo

logger = logging.getLogger(__name__)


class ResponseNode:
    async def __call__(self, state: AgentState) -> AgentState:
        logger.info("entering node=response")
        workflow_trace = append_trace(state, "response")
        sources = [
            GroundingSource(
                source=str(item.get("source", "unknown")),
                score=float(item.get("score", 0.0)),
                snippet=str(item.get("snippet", "")),
            )
            for item in state.get("retrieved_chunks", [])
        ]
        response = ChatResponse(
            session_id=state["session_id"],
            mode=state.get("interaction_mode", "education"),
            message=state.get("message_draft") or "Let us learn one small thing together.",
            follow_up_question=state.get("follow_up_question"),
            topic=state.get("current_topic") or state.get("pending_topic"),
            react=ReactInfo(
                decision=state.get("react_decision", "respond_directly"),
                selected_act=state.get("selected_act", "direct"),
                tool_name=state.get("selected_tool"),
                tool_success=bool(state.get("tool_success")),
                reason=state.get("route_reason", ""),
            ),
            grounding=GroundingInfo(
                used_rag=bool(sources),
                sources=sources,
            ),
            memory=MemoryInfo(
                session_updated=bool(state.get("memory_session_updated")),
                profile_updated=bool(state.get("memory_profile_updated")),
                written_types=list(state.get("memory_written_types", [])),
                consolidated_count=int(state.get("memory_consolidated_count", 0)),
                forgotten_count=int(state.get("memory_forgotten_count", 0)),
            ),
            metadata=ChatMetadata(
                source_mode=state.get("source_mode", "llm"),
                confidence=state.get("confidence", "medium"),
                safety_notes=state.get("safety_notes", ""),
                used_image=bool(state.get("image_base64") or state.get("image_url")),
                dialogue_stage="responded",
                workflow_trace=workflow_trace,
                input_modality=state.get("input_modality", "text"),
                route_reason=state.get("route_reason", ""),
                rag_enabled=bool(state.get("rag_enabled", True)),
            ),
        )
        return {
            "final_response": response.model_dump(),
            "dialogue_stage": "responded",
            "workflow_trace": workflow_trace,
        }
