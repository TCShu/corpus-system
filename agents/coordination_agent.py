"""
CoordinationAgent — LangGraph supervisor that routes natural language queries
to specialist AI agents via message passing.

Graph flow:
  START → router → [frequency | kwic | ngram | keyword | conversational | dynamic] → validator → END
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from agents._shared import get_model
from agents.frequency_agent import FrequencyAgent
from agents.keyword_agent import KeywordAgent
from agents.kwic_agent import KWICAgent
from agents.ngram_agent import NgramAgent
from agents.validation_agent import ValidationAgent
from services.code_execution_service import SafeCodeExecutionService


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    tokens: list[str]
    reference_tokens: list[str] | None
    corpus_text: str
    # routing decision made by the router node
    route: str
    # extracted parameters for specialist agents
    params: dict[str, Any]
    # result from specialist agent
    result: dict[str, Any] | None
    # validation report
    validation: dict[str, Any]
    # final safe flag
    safe: bool
    # conversational reply (when out of scope or chat)
    reply: str | None
    # message history passed between nodes
    messages: list[BaseMessage]


# ---------------------------------------------------------------------------
# Router prompt & node
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """You are the Routing Agent for ACAS (Academic Corpus Analysis System).
Your job is to classify the user's query into exactly one of these categories:

- frequency     : Word frequency / most common words
- kwic          : Keyword-in-context / concordance lines / occurrences of a word
- ngram         : N-grams / bigrams / trigrams / collocations
- keyword       : Keyword comparison / keyness / comparing two corpora
- conversational: Greetings, help requests, questions about ACAS capabilities
- out_of_scope  : Anything unrelated to corpus linguistic analysis

Respond with ONLY the category label (one word, lowercase). No explanation."""

ROUTER_EXAMPLES = """Examples:
"show me the most frequent words" → frequency
"find concordances for 'justice'" → kwic
"what are the top bigrams?" → ngram
"compare keywords with the reference corpus" → keyword
"hello" → conversational
"what can you do?" → conversational
"write me a poem" → out_of_scope
"what's the weather?" → out_of_scope"""

CONVERSATIONAL_SYSTEM = """You are ACAS — the Academic Corpus Analysis System.
You help linguists and researchers analyse text corpora.

You can perform:
• Frequency analysis — most common words and their distribution
• KWIC (Keyword-in-Context) — concordance lines showing a word in context
• N-gram / Collocation analysis — frequent word combinations with PMI scores
• Keyword comparison — identifying distinctive words vs a reference corpus

For greetings and capability questions, respond warmly and helpfully.
For requests outside corpus analysis (poems, code, general chat), politely redirect the user."""

PARAM_EXTRACT_SYSTEM = """You are a parameter extraction assistant. Given a user's corpus analysis query,
extract the relevant parameters as a JSON object.

For frequency: {"top_k": <int, default 20>, "exclude_stopwords": <bool, default true>}
For kwic: {"keyword": "<word to search>", "window_size": <int, default 5>, "max_results": <int, default 50>}
For ngram: {"n_size": <int, default 2>, "min_frequency": <int, default 2>, "top_k": <int, default 20>}
For keyword: {"top_k": <int, default 20>, "min_frequency": <int, default 2>}

Return ONLY the JSON object, no explanation."""


def _build_router_node(llm: ChatOllama):
    def router_node(state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content=ROUTER_SYSTEM + "\n\n" + ROUTER_EXAMPLES),
            HumanMessage(content=state["query"]),
        ]
        response = llm.invoke(messages)
        route = response.content.strip().lower()

        # Normalise to known routes
        valid_routes = {"frequency", "kwic", "ngram", "keyword", "conversational", "out_of_scope"}
        if route not in valid_routes:
            # Let the LLM's intent guide a best-guess
            for candidate in valid_routes:
                if candidate in route:
                    route = candidate
                    break
            else:
                route = "out_of_scope"

        # Extract parameters with a second LLM call for analysis routes
        params: dict[str, Any] = {}
        if route in {"frequency", "kwic", "ngram", "keyword"}:
            param_messages = [
                SystemMessage(content=PARAM_EXTRACT_SYSTEM),
                HumanMessage(content=f"Route: {route}\nQuery: {state['query']}"),
            ]
            param_response = llm.invoke(param_messages)
            try:
                raw = param_response.content.strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    params = json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                params = {}

        new_messages = list(state["messages"]) + [
            HumanMessage(content=state["query"]),
            AIMessage(content=f"[Router] Classified as: {route}. Params: {json.dumps(params)}"),
        ]
        return {**state, "route": route, "params": params, "messages": new_messages}

    return router_node


# ---------------------------------------------------------------------------
# Specialist agent nodes
# ---------------------------------------------------------------------------

def _build_frequency_node(agent: FrequencyAgent):
    def frequency_node(state: AgentState) -> AgentState:
        params = state.get("params", {})
        result = agent.analyze(
            tokens=state["tokens"],
            top_k=params.get("top_k", 20),
            exclude_stopwords=params.get("exclude_stopwords", True),
        )
        new_messages = list(state["messages"]) + [
            AIMessage(content=f"[FrequencyAgent] Analysis complete. Found {len(result.get('rows', []))} rows."),
        ]
        return {**state, "result": result, "messages": new_messages}
    return frequency_node


def _build_kwic_node(agent: KWICAgent):
    def kwic_node(state: AgentState) -> AgentState:
        params = state.get("params", {})
        keyword = params.get("keyword", "")
        if not keyword:
            # Fallback: extract last quoted word or last word from query
            quoted = re.search(r"['\"]([^'\"]+)['\"]", state["query"])
            keyword = quoted.group(1) if quoted else re.findall(r"[a-zA-Z']+", state["query"])[-1]
        result = agent.analyze(
            tokens=state["tokens"],
            keyword=keyword,
            window_size=params.get("window_size", 5),
            max_results=params.get("max_results", 50),
        )
        new_messages = list(state["messages"]) + [
            AIMessage(content=f"[KWICAgent] Found {len(result.get('matches', []))} concordance lines for '{keyword}'."),
        ]
        return {**state, "result": result, "messages": new_messages}
    return kwic_node


def _build_ngram_node(agent: NgramAgent):
    def ngram_node(state: AgentState) -> AgentState:
        params = state.get("params", {})
        result = agent.analyze(
            tokens=state["tokens"],
            n_size=params.get("n_size", 2),
            min_frequency=params.get("min_frequency", 2),
            top_k=params.get("top_k", 20),
        )
        new_messages = list(state["messages"]) + [
            AIMessage(content=f"[NgramAgent] Found {len(result.get('rows', []))} collocations."),
        ]
        return {**state, "result": result, "messages": new_messages}
    return ngram_node


def _build_keyword_node(agent: KeywordAgent):
    def keyword_node(state: AgentState) -> AgentState:
        if not state.get("reference_tokens"):
            result = None
            validation = {
                "safe": False,
                "issues": ["Keyword comparison requires a reference corpus. Please select one."],
                "warnings": [],
            }
            return {**state, "result": result, "validation": validation, "safe": False}

        params = state.get("params", {})
        result = agent.analyze(
            target_tokens=state["tokens"],
            reference_tokens=state["reference_tokens"],
            top_k=params.get("top_k", 20),
            min_frequency=params.get("min_frequency", 2),
        )
        new_messages = list(state["messages"]) + [
            AIMessage(content=f"[KeywordAgent] Comparison complete. {len(result.get('rows', []))} keywords found."),
        ]
        return {**state, "result": result, "messages": new_messages}
    return keyword_node


def _build_conversational_node(llm: ChatOllama):
    def conversational_node(state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content=CONVERSATIONAL_SYSTEM),
            HumanMessage(content=state["query"]),
        ]
        response = llm.invoke(messages)
        reply = response.content.strip()
        new_messages = list(state["messages"]) + [
            AIMessage(content=f"[ConversationalAgent] {reply}"),
        ]
        return {**state, "result": None, "reply": reply, "safe": True, "messages": new_messages}
    return conversational_node


def _build_out_of_scope_node(llm: ChatOllama):
    def out_of_scope_node(state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content=CONVERSATIONAL_SYSTEM),
            HumanMessage(content=(
                f"The user asked: '{state['query']}'\n\n"
                "This is outside the scope of corpus linguistic analysis. "
                "Politely explain what ACAS is designed for and suggest relevant query types they could try instead."
            )),
        ]
        response = llm.invoke(messages)
        reply = response.content.strip()
        new_messages = list(state["messages"]) + [
            AIMessage(content=f"[OutOfScopeAgent] {reply}"),
        ]
        return {**state, "result": None, "reply": reply, "safe": True, "messages": new_messages}
    return out_of_scope_node


# ---------------------------------------------------------------------------
# Validation node
# ---------------------------------------------------------------------------

def _build_validation_node(agent: ValidationAgent):
    def validation_node(state: AgentState) -> AgentState:
        if state.get("result") is None:
            # Conversational / out-of-scope — already handled
            return state
        validation = agent.validate_result(state["result"])
        new_messages = list(state["messages"]) + [
            AIMessage(content=f"[ValidationAgent] Safe={validation['safe']}. Issues: {validation.get('issues', [])}."),
        ]
        return {**state, "validation": validation, "safe": validation["safe"], "messages": new_messages}
    return validation_node


# ---------------------------------------------------------------------------
# Routing edge function
# ---------------------------------------------------------------------------

def _route_edge(state: AgentState) -> Literal["frequency", "kwic", "ngram", "keyword", "conversational", "out_of_scope"]:
    return state["route"]


# ---------------------------------------------------------------------------
# CoordinationAgent
# ---------------------------------------------------------------------------

class CoordinationAgent:
    """LangGraph supervisor that routes queries to specialist AI agents."""

    def __init__(self) -> None:
        llm = ChatOllama(model=get_model())

        self._frequency_agent = FrequencyAgent()
        self._kwic_agent = KWICAgent()
        self._ngram_agent = NgramAgent()
        self._keyword_agent = KeywordAgent()
        self._validation_agent = ValidationAgent()
        self._code_runner = SafeCodeExecutionService()

        # Build graph
        graph = StateGraph(AgentState)

        graph.add_node("router", _build_router_node(llm))
        graph.add_node("frequency", _build_frequency_node(self._frequency_agent))
        graph.add_node("kwic", _build_kwic_node(self._kwic_agent))
        graph.add_node("ngram", _build_ngram_node(self._ngram_agent))
        graph.add_node("keyword", _build_keyword_node(self._keyword_agent))
        graph.add_node("conversational", _build_conversational_node(llm))
        graph.add_node("out_of_scope", _build_out_of_scope_node(llm))
        graph.add_node("validator", _build_validation_node(self._validation_agent))

        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", _route_edge)
        graph.add_edge("frequency", "validator")
        graph.add_edge("kwic", "validator")
        graph.add_edge("ngram", "validator")
        graph.add_edge("keyword", "validator")
        graph.add_edge("conversational", END)
        graph.add_edge("out_of_scope", END)
        graph.add_edge("validator", END)

        self._graph = graph.compile()

    def route_query(self, query: str) -> str:
        """Return only the routing decision for a query (no specialist execution)."""
        llm = ChatOllama(model=get_model())
        messages = [
            SystemMessage(content=ROUTER_SYSTEM + "\n\n" + ROUTER_EXAMPLES),
            HumanMessage(content=query),
        ]
        response = llm.invoke(messages)
        route = response.content.strip().lower()
        valid_routes = {"frequency", "kwic", "ngram", "keyword", "conversational", "out_of_scope"}
        if route not in valid_routes:
            for candidate in valid_routes:
                if candidate in route:
                    return candidate
            return "out_of_scope"
        return route

    def execute(
        self,
        query: str,
        tokens: list[str],
        reference_tokens: list[str] | None = None,
        corpus_text: str = "",
    ) -> dict[str, Any]:
        initial_state: AgentState = {
            "query": query,
            "tokens": tokens,
            "reference_tokens": reference_tokens,
            "corpus_text": corpus_text,
            "route": "",
            "params": {},
            "result": None,
            "validation": {"safe": True, "issues": [], "warnings": []},
            "safe": True,
            "reply": None,
            "messages": [],
        }

        final_state = self._graph.invoke(initial_state)

        # Conversational / out-of-scope: return reply as result
        if final_state.get("reply"):
            return {
                "route": final_state["route"],
                "safe": True,
                "result": {
                    "analysis_type": "conversational",
                    "reply": final_state["reply"],
                },
                "validation": {"safe": True, "issues": [], "warnings": []},
            }

        return {
            "route": final_state["route"],
            "safe": final_state["safe"],
            "result": final_state["result"],
            "validation": final_state["validation"],
        }
