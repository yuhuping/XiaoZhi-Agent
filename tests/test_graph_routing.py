from __future__ import annotations

from app.agent.graph import route_by_mode


class TestRouteByMode:
    def test_education_routes_to_plan_execute(self) -> None:
        state = {"interaction_mode": "education"}
        assert route_by_mode(state) == "plan_execute"

    def test_parent_routes_to_react(self) -> None:
        state = {"interaction_mode": "parent"}
        assert route_by_mode(state) == "react"

    def test_companion_routes_to_react(self) -> None:
        state = {"interaction_mode": "companion"}
        assert route_by_mode(state) == "react"
