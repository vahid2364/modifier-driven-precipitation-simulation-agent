"""
Conductor: LangGraph StateGraph that routes between agents.

Graph topology (Phase 1):
    START → scout → sim_runner → [error_tracer → sim_runner (retry)] → analyst → bo → END

Phase 2 (BO active):
    analyst → bo → scout  (loop until convergence or budget exhausted)
"""

from langgraph.graph import StateGraph, END
from state import SimState
from src.agents import ScoutAgent, SimRunnerAgent, AnalystAgent, BOAgent, ErrorTracerAgent, BeamerAgent

MAX_RETRIES = 2   # max ErrorTracer → SimRunner retry cycles


def _scout_node(state: SimState) -> SimState:
    return ScoutAgent().run(state)


def _sim_runner_node(state: SimState) -> SimState:
    state.setdefault("_sim_retries", 0)
    return SimRunnerAgent().run(state)


def _error_tracer_node(state: SimState) -> SimState:
    return ErrorTracerAgent().run(state)


def _analyst_node(state: SimState) -> SimState:
    return AnalystAgent().run(state)


def _bo_node(state: SimState) -> SimState:
    return BOAgent().run(state)


def _beamer_node(state: SimState) -> SimState:
    return BeamerAgent().run(state)


def _route_after_scout(state: SimState) -> str:
    if state.get("error"):
        print(f"[Conductor] ScoutAgent failed: {state['error']}")
        return END
    return "sim_runner"


def _route_after_sim_runner(state: SimState) -> str:
    if state.get("error"):
        retries = state.get("_sim_retries", 0)
        if retries < MAX_RETRIES:
            print(f"[Conductor] SimRunner failed — attempting ErrorTracer fix "
                  f"(attempt {retries + 1}/{MAX_RETRIES})")
            state["_sim_retries"] = retries + 1
            return "error_tracer"
        print(f"[Conductor] SimRunner failed after {MAX_RETRIES} retries — halting.")
        return END
    return "analyst"


def _route_after_error_tracer(state: SimState) -> str:
    """If ErrorTracer cleared the error, retry SimRunner. Otherwise halt."""
    if state.get("error"):
        print(f"[Conductor] ErrorTracer could not fix error — halting.")
        return END
    return "sim_runner"


def _route_after_analyst(state: SimState) -> str:
    if state.get("error"):
        print(f"[Conductor] AnalystAgent failed: {state['error']}")
        return END
    return "bo"


def _route_after_bo(state: SimState) -> str:
    """Phase 1: go to beamer. Phase 2: loop if BO proposes a next modifier."""
    if state.get("next_modifier"):
        return "scout"
    return "beamer"


def build_conductor() -> StateGraph:
    graph = StateGraph(SimState)

    graph.add_node("scout",        _scout_node)
    graph.add_node("sim_runner",   _sim_runner_node)
    graph.add_node("error_tracer", _error_tracer_node)
    graph.add_node("analyst",      _analyst_node)
    graph.add_node("bo",           _bo_node)
    graph.add_node("beamer",       _beamer_node)

    graph.set_entry_point("scout")

    graph.add_conditional_edges("scout", _route_after_scout, {
        "sim_runner": "sim_runner", END: END,
    })
    graph.add_conditional_edges("sim_runner", _route_after_sim_runner, {
        "analyst":     "analyst",
        "error_tracer": "error_tracer",
        END:            END,
    })
    graph.add_conditional_edges("error_tracer", _route_after_error_tracer, {
        "sim_runner": "sim_runner", END: END,
    })
    graph.add_conditional_edges("analyst", _route_after_analyst, {
        "bo": "bo", END: END,
    })
    graph.add_conditional_edges("bo", _route_after_bo, {
        "scout": "scout", "beamer": "beamer",
    })
    graph.add_edge("beamer", END)

    return graph.compile()
