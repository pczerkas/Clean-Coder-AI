import os
from typing import Sequence, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.tools import tool
from langchain.tools.render import render_text_description
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_experimental.llms import ChatLlamaAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai.chat_models import ChatOpenAI
from langchain_together import ChatTogether
from langgraph.graph import StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from llamaapi import LlamaAPI

from models.codeium_chat import CodeiumChatModel
from rag.retrieval import vdb_availabe
from tools.tools import (
    list_dir,
    retrieve_files_by_semantic_query,
    see_file,
    see_image,
    tool_description_end,
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
    set_docstring,
)

load_dotenv(find_dotenv(), override=True)
mistral_api_key = os.getenv("MISTRAL_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")


@tool
@set_docstring(
    f"""That tool outputs message to programmer as well as list of files executor will need to change.
    Use that tool only when you 100% sure you found all the files Executor will need to modify.
    If not, do additional research.
    Include only the files you are convinced that will be useful.

    tool input:
    :param message_to_programmer: "Programmer: what do you need to do"],
    :param files_to_work_on: ["List", "of", "existing files", "to potentially introduce", "changes"],
    :param reference_files: ["List", "of code files", "useful to code reference"],
{tool_description_end}"""
)
def final_response(message_to_programmer, files_to_work_on, reference_files):
    pass


tools = [list_dir, see_file, final_response]
if vdb_availabe:
    tools.append(retrieve_files_by_semantic_query)
rendered_tools = render_text_description(tools)

# stop_sequence = "\n```\n"
stop_sequence = None

# original LLM
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.2,
# )

# llm = ChatAnthropic(
#     model='claude-3-5-sonnet-20240620',
#     temperature=0.2,
# )

# llm = ChatGroq(model="llama3-70b-8192", temperature=0.3).with_config({"run_name": "Researcher"})
# llm = ChatOllama(model="llama3.1")
# llm = ChatMistralAI(api_key=mistral_api_key, model="mistral-large-latest")
# llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", temperature=0.3).with_config({"run_name": "Researcher"})
# llm = ChatNVIDIA(model="nvidia/llama3-chatqa-1.5-70b")
# llama = LlamaAPI(os.getenv("LLAMA_API_KEY"))
# llm = ChatLlamaAPI(client=llama)

# llm = ChatOpenAI(
#     #model='deepseek-chat',
#     model='deepseek-coder',
#     openai_api_key=deepseek_api_key,
#     #openai_api_base='https://api.deepseek.com/v1',
#     openai_api_base='https://api.deepseek.com/beta',
#     temperature=0.0,
#     max_tokens=8000
# )
# llm = ChatGroq(
#     #model="llama3-70b-8192", # too many requests (tokens)
#     #model="llama-3.1-405b-reasoning", # 404 model does not exists
#     #model="llama-3.1-70b-versatile",
#     model="Mixtral-8x7b-32768", # stupid
#     temperature=0.3,
# )
# llm = CodeiumChatModel(
#     workspace_id='file_home_przemek_VSCode_20Projects_X_code_workspace',
#     responses=['a','b'],
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    # convert_system_message_to_human=True,
)


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


bad_json_format_msg = (
    "Bad json format. Json should contain fields 'tool' and 'tool_input' "
    "and enclosed with '```json', '```' tags."
)

project_knowledge = read_project_knowledge()
tool_executor = ToolExecutor(tools)
system_message_content = f"""As a curious filesystem researcher, examine files thoroughly, prioritizing comprehensive checks.
You checking a lot of different folders looking around for interesting files (hey, you are very curious!) before giving the final answer.
The more folders/files you will check, the more they will pay you.
When you discover significant dependencies from one file to another, ensure to inspect both.
Your final selection should include files needed to be modified or needed as reference for a programmer
(to see how code in similar file is implemented - it is important).
Avoid recommending unseen or non-existent files in final response. Start from '/' directory.
Avoid checking files, that you have already checked.
You need to POINT OUT ALL FILES programmer needed to see to execute task.

Task is:
'''
{{task}}
'''
As a researcher, you are not allowed to make any code modifications.

Knowledge about project (not so important):
{project_knowledge}

You have access to following tools:
{rendered_tools}

First, provide step by step reasoning about results of your previous action. Think what do you need to find now in order to accomplish the task.
Next, generate response using json template: Choose only one tool to use.
```json
{{{{
    "tool": "$TOOL_NAME",
    "tool_input": "$TOOL_PARAMS",
}}}}
```
"""


# node functions
def call_model_researcher(state):
    state, response = call_model(state, llm, stop_sequence_to_add=stop_sequence)
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

researcher_workflow.add_node("agent", call_model_researcher)
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


def research_task(task):
    print("Researcher starting its work")
    system_message = system_message_content.format(task=task)
    inputs = {
        "messages": [SystemMessage(content=system_message), HumanMessage(content=f"Go")]
    }
    researcher_response = researcher.invoke(inputs, {"recursion_limit": 500})[
        "messages"
    ][-2]

    # tool_json = find_tool_xml(researcher_response.content)
    tool_json = find_tool_json(researcher_response.content)
    message_to_programmer = tool_json["tool_input"]["message_to_programmer"]
    text_files = set(
        tool_json["tool_input"]["files_to_work_on"]
        + tool_json["tool_input"]["reference_files"]
    )
    files_contents = check_files_contents(text_files)

    return message_to_programmer, text_files, files_contents
