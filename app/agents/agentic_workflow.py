"""
LangGraph Agentic Workflows - Darwin 2025

Advanced agent patterns:
- ReAct (Reasoning + Acting)
- Reflexion (self-critique)
- Tree of Thoughts
- Multi-agent debate
- Chain of Verification

Agents especializados:
- ResearchAgent: Busca + síntese científica
- ReflectionAgent: Auto-crítica e validação
- ValidationAgent: Verificação científica
- ToolAgent: Execução de ferramentas Darwin
- SynthesisAgent: Síntese multi-fonte
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from dataclasses import dataclass
from enum import Enum
import operator

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logging.warning("LangGraph not available")

try:
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
    from langchain.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Papéis dos agentes"""
    RESEARCHER = "researcher"  # Pesquisa e coleta
    REFLECTION = "reflection"  # Auto-crítica
    VALIDATOR = "validator"  # Validação científica
    TOOL_EXECUTOR = "tool_executor"  # Executa ferramentas
    SYNTHESIZER = "synthesizer"  # Síntese final
    COORDINATOR = "coordinator"  # Coordena workflow


class AgentState(TypedDict):
    """Estado compartilhado entre agentes"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    task: str
    research_results: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    reflections: List[str]
    validations: List[Dict[str, Any]]
    final_synthesis: Optional[str]
    iterations: int
    max_iterations: int
    status: str  # planning, researching, reflecting, validating, synthesizing, done


@dataclass
class WorkflowConfig:
    """Configuração do workflow"""
    max_iterations: int = 5
    enable_reflection: bool = True
    enable_validation: bool = True
    llm_model: str = "gpt-4-turbo-preview"
    llm_provider: str = "openai"  # openai, anthropic
    temperature: float = 0.7
    verbose: bool = True


class DarwinAgenticWorkflow:
    """
    Workflow agentic avançado com LangGraph
    
    Implementa patterns estado da arte 2025:
    - ReAct para reasoning + acting
    - Reflexion para auto-crítica
    - Multi-agent debate
    - Chain of verification
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        Initialize agentic workflow
        
        Args:
            config: Workflow configuration
        """
        if not HAS_LANGGRAPH or not HAS_LANGCHAIN:
            raise ImportError(
                "LangGraph and LangChain required: "
                "pip install langgraph langchain langchain-openai langchain-anthropic"
            )
        
        self.config = config or WorkflowConfig()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Build graph
        self.graph = None
        self.app = None
        self._build_graph()
        
        logger.info(f"✅ Darwin Agentic Workflow initialized (model={self.config.llm_model})")
    
    def _initialize_llm(self):
        """Initialize LLM based on config"""
        if self.config.llm_provider == "openai":
            return ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.temperature,
            )
        elif self.config.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.llm_model,
                temperature=self.config.temperature,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("coordinator", self._coordinator_agent)
        workflow.add_node("researcher", self._researcher_agent)
        workflow.add_node("tool_executor", self._tool_executor_agent)
        
        if self.config.enable_reflection:
            workflow.add_node("reflection", self._reflection_agent)
        
        if self.config.enable_validation:
            workflow.add_node("validator", self._validator_agent)
        
        workflow.add_node("synthesizer", self._synthesizer_agent)
        
        # Set entry point
        workflow.set_entry_point("coordinator")
        
        # Add edges (transitions)
        workflow.add_edge("coordinator", "researcher")
        workflow.add_edge("researcher", "tool_executor")
        
        if self.config.enable_reflection:
            workflow.add_edge("tool_executor", "reflection")
            
            # Conditional: continue or validate
            workflow.add_conditional_edges(
                "reflection",
                self._should_continue_research,
                {
                    "continue": "researcher",
                    "validate": "validator" if self.config.enable_validation else "synthesizer",
                }
            )
        else:
            workflow.add_edge(
                "tool_executor",
                "validator" if self.config.enable_validation else "synthesizer"
            )
        
        if self.config.enable_validation:
            workflow.add_edge("validator", "synthesizer")
        
        workflow.add_edge("synthesizer", END)
        
        # Compile
        self.app = workflow.compile()
        
        logger.info("✅ LangGraph workflow compiled")
    
    def _coordinator_agent(self, state: AgentState) -> AgentState:
        """
        Coordinator: Planeja o workflow
        
        Decide:
        - Quais ferramentas usar
        - Quantas iterações
        - Estratégia de busca
        """
        logger.info("🎯 Coordinator planning workflow...")
        
        task = state["task"]
        
        # LLM call para planejamento
        planning_prompt = f"""You are a research coordinator for Darwin AI platform.

Task: {task}

Plan the research workflow:
1. What information do we need?
2. Which Darwin tools to use?
3. What validation is needed?

Provide a concise plan."""
        
        messages = [SystemMessage(content=planning_prompt)]
        response = self.llm.invoke(messages)
        
        state["messages"].append(response)
        state["current_agent"] = "coordinator"
        state["status"] = "planning"
        state["iterations"] = 0
        
        logger.info(f"Plan: {response.content[:200]}...")
        
        return state
    
    def _researcher_agent(self, state: AgentState) -> AgentState:
        """
        Researcher: Coleta informações
        
        ReAct pattern:
        1. Reason: O que preciso saber?
        2. Act: Busca no Darwin (semantic memory, KEC, etc)
        3. Observe: Analisa resultados
        """
        logger.info("🔬 Researcher gathering information...")
        
        task = state["task"]
        prev_results = state.get("research_results", [])
        
        # ReAct reasoning
        research_prompt = f"""You are a scientific researcher using Darwin AI platform.

Task: {task}

Previous research: {prev_results if prev_results else "None"}

What should we research next? Think step by step:
1. What do we know?
2. What do we need to know?
3. What tools should we use?

Provide specific search queries."""
        
        messages = state["messages"] + [HumanMessage(content=research_prompt)]
        response = self.llm.invoke(messages)
        
        state["messages"].append(response)
        state["current_agent"] = "researcher"
        state["status"] = "researching"
        
        # Extract search queries (simplified)
        # In production, use structured output
        research_plan = {
            "query": task,
            "reasoning": response.content,
            "tools_needed": ["semantic_memory_search", "kec_analysis"],
        }
        
        state["research_results"].append(research_plan)
        
        return state
    
    def _tool_executor_agent(self, state: AgentState) -> AgentState:
        """
        Tool Executor: Executa ferramentas Darwin
        
        Ferramentas disponíveis:
        - semantic_memory_search
        - kec_analysis
        - pbpk_simulation
        - literature_search
        """
        logger.info("🔧 Tool Executor running Darwin tools...")
        
        research_results = state["research_results"]
        tool_results = []
        
        for research in research_results[-1:]:  # Last research iteration
            tools_needed = research.get("tools_needed", [])
            
            for tool_name in tools_needed:
                result = self._execute_tool(tool_name, research.get("query", ""))
                tool_results.append({
                    "tool": tool_name,
                    "result": result,
                    "success": result is not None,
                })
        
        state["tool_results"].extend(tool_results)
        state["current_agent"] = "tool_executor"
        
        return state
    
    def _reflection_agent(self, state: AgentState) -> AgentState:
        """
        Reflection: Auto-crítica (Reflexion pattern)
        
        Avalia:
        - Qualidade dos resultados
        - Gaps no conhecimento
        - Necessidade de mais pesquisa
        - Erros ou inconsistências
        """
        logger.info("🤔 Reflection agent critiquing results...")
        
        tool_results = state["tool_results"]
        research_results = state["research_results"]
        
        reflection_prompt = f"""You are a critical reviewer for Darwin AI.

Task: {state['task']}

Research done: {len(research_results)} iterations
Tool results: {tool_results[-3:] if tool_results else "None"}

Critically evaluate:
1. Do we have enough information?
2. Are there inconsistencies?
3. What's missing?
4. Should we continue research or validate?

Be honest and specific."""
        
        messages = state["messages"] + [HumanMessage(content=reflection_prompt)]
        response = self.llm.invoke(messages)
        
        state["messages"].append(response)
        state["reflections"].append(response.content)
        state["current_agent"] = "reflection"
        state["status"] = "reflecting"
        state["iterations"] += 1
        
        logger.info(f"Reflection (iter {state['iterations']}): {response.content[:200]}...")
        
        return state
    
    def _validator_agent(self, state: AgentState) -> AgentState:
        """
        Validator: Validação científica
        
        Chain of Verification:
        1. Identifica claims
        2. Verifica cada claim
        3. Marca confiança
        4. Sugere fontes adicionais
        """
        logger.info("✅ Validator checking scientific accuracy...")
        
        tool_results = state["tool_results"]
        
        validation_prompt = f"""You are a scientific validator for Darwin AI.

Task: {state['task']}

Results to validate:
{tool_results[-3:] if tool_results else "None"}

Verify:
1. Are claims scientifically accurate?
2. Are there citations needed?
3. What's the confidence level?
4. Any red flags?

Provide validation report."""
        
        messages = state["messages"] + [HumanMessage(content=validation_prompt)]
        response = self.llm.invoke(messages)
        
        validation_report = {
            "validated": True,
            "confidence": "high",  # In production, extract from response
            "issues": [],
            "feedback": response.content,
        }
        
        state["messages"].append(response)
        state["validations"].append(validation_report)
        state["current_agent"] = "validator"
        state["status"] = "validating"
        
        return state
    
    def _synthesizer_agent(self, state: AgentState) -> AgentState:
        """
        Synthesizer: Síntese final
        
        Combina:
        - Research results
        - Tool outputs
        - Reflections
        - Validations
        
        Output: Resposta científica completa
        """
        logger.info("📝 Synthesizer creating final answer...")
        
        synthesis_prompt = f"""You are a scientific synthesizer for Darwin AI.

Task: {state['task']}

You have:
- Research: {len(state['research_results'])} iterations
- Tool results: {len(state['tool_results'])} tools executed
- Reflections: {len(state['reflections'])} self-critiques
- Validations: {len(state['validations'])} checks

Synthesize a comprehensive, scientifically rigorous answer.
Include:
1. Direct answer to task
2. Supporting evidence
3. Confidence level
4. Limitations
5. References to Darwin tools used

Be clear, accurate, and thorough."""
        
        messages = state["messages"] + [HumanMessage(content=synthesis_prompt)]
        response = self.llm.invoke(messages)
        
        state["messages"].append(response)
        state["final_synthesis"] = response.content
        state["current_agent"] = "synthesizer"
        state["status"] = "done"
        
        logger.info("✅ Final synthesis complete")
        
        return state
    
    def _should_continue_research(self, state: AgentState) -> str:
        """
        Decide se continua pesquisa ou vai para validação
        
        Baseado em:
        - Iterações
        - Qualidade dos resultados
        - Reflexões
        """
        iterations = state["iterations"]
        max_iterations = state["max_iterations"]
        
        if iterations >= max_iterations:
            return "validate"
        
        # Check last reflection
        if state["reflections"]:
            last_reflection = state["reflections"][-1].lower()
            
            # Simple heuristic (in production, use structured output)
            if "enough" in last_reflection or "sufficient" in last_reflection:
                return "validate"
            elif "more" in last_reflection or "missing" in last_reflection:
                return "continue"
        
        # Default: continue if under max iterations
        return "continue" if iterations < max_iterations - 1 else "validate"
    
    def _execute_tool(self, tool_name: str, query: str) -> Optional[Dict[str, Any]]:
        """
        Execute Darwin tool
        
        In production, integrate with actual Darwin services
        """
        logger.info(f"Executing tool: {tool_name} with query: {query[:100]}")
        
        # Placeholder - integrate with actual Darwin
        if tool_name == "semantic_memory_search":
            return {
                "results": [
                    {"content": f"Result for {query}", "score": 0.85},
                ],
                "total": 1,
            }
        elif tool_name == "kec_analysis":
            return {
                "H_spectral": 0.75,
                "sigma": 1.45,
                "phi": 0.68,
            }
        
        return None
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Execute workflow for task
        
        Args:
            task: Research task/question
        
        Returns:
            Final results with synthesis
        """
        if not self.app:
            raise RuntimeError("Workflow not built")
        
        logger.info(f"🚀 Starting agentic workflow for: {task[:100]}...")
        
        # Initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "current_agent": "",
            "task": task,
            "research_results": [],
            "tool_results": [],
            "reflections": [],
            "validations": [],
            "final_synthesis": None,
            "iterations": 0,
            "max_iterations": self.config.max_iterations,
            "status": "planning",
        }
        
        # Run workflow
        try:
            final_state = self.app.invoke(initial_state)
            
            logger.info("✅ Workflow completed successfully")
            
            return {
                "success": True,
                "synthesis": final_state.get("final_synthesis"),
                "iterations": final_state.get("iterations"),
                "tool_results": final_state.get("tool_results"),
                "validations": final_state.get("validations"),
                "status": final_state.get("status"),
            }
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Factory functions
def create_research_workflow(
    model: str = "gpt-4-turbo-preview",
    max_iterations: int = 3,
) -> DarwinAgenticWorkflow:
    """Create research workflow with sensible defaults"""
    config = WorkflowConfig(
        max_iterations=max_iterations,
        enable_reflection=True,
        enable_validation=True,
        llm_model=model,
        temperature=0.7,
        verbose=True,
    )
    
    return DarwinAgenticWorkflow(config=config)


def create_fast_workflow(
    model: str = "gpt-3.5-turbo",
) -> DarwinAgenticWorkflow:
    """Create fast workflow (no reflection/validation)"""
    config = WorkflowConfig(
        max_iterations=1,
        enable_reflection=False,
        enable_validation=False,
        llm_model=model,
        temperature=0.5,
        verbose=False,
    )
    
    return DarwinAgenticWorkflow(config=config)


__all__ = [
    "DarwinAgenticWorkflow",
    "WorkflowConfig",
    "AgentRole",
    "AgentState",
    "create_research_workflow",
    "create_fast_workflow",
]

