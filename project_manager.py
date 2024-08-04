from typing import Sequence, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.tools.render import render_text_description
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

from tools.tools_project_manager import (
    add_task,
    ask_programmer_to_execute_task,
    ask_tester_to_check_if_change_been_implemented_correctly,
    delete_task,
    get_project_tasks,
    mark_task_as_done,
    modify_task,
)
from utilities.langgraph_common_functions import (
    after_ask_human_condition,
    ask_human,
    call_model,
    call_tool,
)
from utilities.util_functions import (
    check_files_contents,
    find_tool_json,
    find_tool_xml,
    print_wrapped,
    read_project_knowledge,
)

load_dotenv(find_dotenv(), override=True)
tools = [
    get_project_tasks,
    add_task,
    modify_task,
    delete_task,
    mark_task_as_done,
    ask_programmer_to_execute_task,
    ask_tester_to_check_if_change_been_implemented_correctly,
]
rendered_tools = render_text_description(tools)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.4,
).with_config({"run_name": "Planer"})


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


bad_json_format_msg = (
    "Bad json format. Json should contain fields 'tool' and 'tool_input' "
    "and enclosed with '```json', '```' tags."
)

project_description = read_project_knowledge()
tool_executor = ToolExecutor(tools)

project_description = """Backend for internet shop with skin bags.
"""

system_message = SystemMessage(
    content=f"""
You are project manager that guides programmer in his work, plan future tasks, checks quality of their execution and
replans over and over (if needed) until project is finished.

Here is description of the project you work on:
{project_description}

You have access to following tools:
{rendered_tools}\n

First, provide step by step reasoning about what do you need to find in order to accomplish the task.
Next, generate response using json template: Choose only one tool to use.
```json
{{
    "tool": "$TOOL_NAME",
    "tool_input": "$TOOL_PARAMS",
}}
```
"""
)


# node functions
def call_model_manager(state):
    state, response = call_model(state, llm)
    # safety mechanism for a bad json
    tool_call = response.tool_call
    if tool_call is None or "tool" not in tool_call:
        state["messages"].append(HumanMessage(content=bad_json_format_msg))
    return state


def call_tool_researcher(state):
    return call_tool(state, tool_executor)


# Logic for conditional edges
def after_agent_condition(state):
    last_message = state["messages"][-1]

    if last_message.content == bad_json_format_msg:
        return "agent"
    elif last_message.tool_call["tool"] == "final_response":
        return "human"
    else:
        return "tool"


# workflow definition
researcher_workflow = StateGraph(AgentState)

researcher_workflow.add_node("agent", call_model_manager)
researcher_workflow.add_node("tool", call_tool_researcher)
researcher_workflow.add_node("human", ask_human)

researcher_workflow.set_entry_point("agent")

researcher_workflow.add_conditional_edges(
    "agent",
    after_agent_condition,
)
researcher_workflow.add_conditional_edges(
    "human",
    after_ask_human_condition,
)
researcher_workflow.add_edge("tool", "agent")

researcher = researcher_workflow.compile()


def research_task():
    print("Manager starting its work")
    inputs = {"messages": [system_message, HumanMessage(content="Go")]}
    researcher_response = researcher.invoke(inputs, {"recursion_limit": 500})[
        "messages"
    ][-2]


if __name__ == "__main__":
    research_task()
