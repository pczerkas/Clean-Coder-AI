import os
from typing import Sequence, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.tools import tool
from langchain.tools.render import render_text_description
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langchain_together import ChatTogether
from langgraph.graph import StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

from tools.tools import (
    WRONG_EXECUTION_WORD,
    ask_human_tool,
    create_directory,
    create_file_with_code,
    insert_code,
    list_dir,
    rename_directory,
    replace_code,
    see_file,
    tool_description_end,
)
from utilities.langgraph_common_functions import (
    after_ask_human_condition,
    ask_human,
    call_model,
    call_tool,
)
from utilities.util_functions import (
    check_application_logs,
    check_file_contents,
    find_tool_json,
    find_tool_xml,
    print_wrapped,
    read_project_knowledge,
    set_docstring,
)

load_dotenv(find_dotenv(), override=True)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
log_file_path = os.getenv("LOG_FILE")


@tool
@set_docstring(
    f"""Call that tool when all changes are implemented to tell the job is done.
    If you have no idea which tool to call, call that.
{tool_description_end}"""
)
def final_response():
    pass


tools = [
    list_dir,
    see_file,
    insert_code,
    replace_code,
    create_file_with_code,
    create_directory,
    rename_directory,
    ask_human_tool,
    final_response,
]
rendered_tools = render_text_description(tools)

stop_sequence = "\n```\n"

# llm = ChatOpenAI(model="gpt-4o", temperature=0).with_config({"run_name": "Executor"})

# llm = ChatAnthropic(
#     model='claude-3-5-sonnet-20240620',
#     temperature=0.2,
#     max_tokens=2000,
#     stop=[stop_sequence]
# ).with_config({"run_name": "Executor"})

# llm = ChatGroq(model="llama3-70b-8192", temperature=0).with_config({"run_name": "Executor"})
# llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", temperature=0).with_config({"run_name": "Executor"})
# llm = ChatOllama(model="mixtral"), temperature=0).with_config({"run_name": "Executor"})

# llm = ChatOpenAI(
#     #model='deepseek-chat',
#     model='deepseek-coder',
#     openai_api_key=deepseek_api_key,
#     #openai_api_base='https://api.deepseek.com/v1',
#     openai_api_base='https://api.deepseek.com/beta',
#     temperature=0.0,
#     max_tokens=8000,
#     stop=[stop_sequence]
# ).with_config({"run_name": "Executor"})
# llm = ChatGroq(
#     #model="llama3-70b-8192", # too many requests (tokens)
#     #model="llama-3.1-405b-reasoning", # 404 model does not exists
#     #model="llama-3.1-70b-versatile",
#     model="Mixtral-8x7b-32768", # stupid
#     temperature=0.3,
# ).with_config({"run_name": "Executor"})
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    # convert_system_message_to_human=True,
).with_config({"run_name": "Executor"})


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


tool_executor = ToolExecutor(tools)
system_message = SystemMessage(
    content=f"""
You are a senior programmer tasked with refining an existing codebase. Your goal is to incrementally
introduce improvements using a set of provided tools. Each change should be implemented step by step,
meaning you make one modification at a time. Focus on enhancing individual functions or lines of code
rather than rewriting entire files at once.
\n\n
Tools to your disposal:\n
{rendered_tools}
\n\n
You have to use tools to tell me what changes need to be made - do not just say that you cannot do something,
because you are AI model.
First, write your thinking process. Think step by step about what do you need to do to accomplish the task.
Reasoning part of your response is very important, never miss it! Even if the next step seems to be obvious.
Next, call tool using template. Use only one json at once! If you want to introduce few changes, just choose one of them;
rest will be possible to do later.
```json
{{
    "tool": "$TOOL_NAME",
    "tool_input": "$TOOL_PARAMS",
}}
```
"""
)

bad_json_format_msg = """Bad json format. Json should be enclosed with '```json', '```' tags.
Code inside of json should be provided in the way that not makes json invalid.
There should be only one json in the response."""


class Executor:
    def __init__(self, files):
        self.files = files

        # workflow definition
        executor_workflow = StateGraph(AgentState)

        executor_workflow.add_node("agent", self.call_model_executor)
        # executor_workflow.add_node("checker", self.call_model_checker)
        executor_workflow.add_node("tool", self.call_tool_executor)
        executor_workflow.add_node("check_log", self.check_log)
        executor_workflow.add_node("human", ask_human)

        executor_workflow.set_entry_point("agent")

        # executor_workflow.add_edge("agent", "checker")
        executor_workflow.add_edge("tool", "agent")
        executor_workflow.add_conditional_edges("agent", self.after_agent_condition)
        executor_workflow.add_conditional_edges(
            "check_log", self.after_check_log_condition
        )
        executor_workflow.add_conditional_edges("human", after_ask_human_condition)

        self.executor = executor_workflow.compile()

    # node functions
    def call_model_executor(self, state):
        # stop_sequence = None
        state, response = call_model(state, llm, stop_sequence_to_add=stop_sequence)
        # safety mechanism for a bad json
        tool_call = response.tool_call
        if tool_call is None or "tool" not in tool_call:
            state["messages"].append(HumanMessage(content=bad_json_format_msg))
            print("\nBad json provided, asked to provide again.")
        elif tool_call == "Multiple jsons found.":
            state["messages"].append(
                HumanMessage(
                    content="You written multiple jsons at once. If you want to execute multiple "
                    "actions, choose only one for now; rest you can execute later."
                )
            )
            print("\nToo many jsons provided, asked to provide one.")
        elif tool_call == "No json found in response.":
            state["messages"].append(
                HumanMessage(
                    content="Good. Please provide a json tool call to execute an action."
                )
            )
            print("\nNo json provided, asked to provide one.")
        return state

    def call_tool_executor(self, state):
        last_ai_message = state["messages"][-1]
        state = call_tool(state, tool_executor)
        if last_ai_message.tool_call["tool"] == "create_file_with_code":
            self.files.add(last_ai_message.tool_call["tool_input"]["filename"])
        if last_ai_message.tool_call["tool"] in [
            "insert_code",
            "replace_code",
            "create_file_with_code",
        ]:
            # marking messages if they haven't introduced changes
            if last_ai_message.content.startswith(WRONG_EXECUTION_WORD):
                # last tool response message
                state["messages"][-1].to_remove = True
                print(
                    "check if to_remove flag saved: ",
                    state["messages"][-1],
                    state["messages"][-1].to_remove,
                )
            else:
                state = self.exchange_file_contents(state)
            print(
                "to_removes: ",
                len([msg for msg in state["messages"] if hasattr(msg, "to_remove")]),
            )
            # checking if we have at least 3 "to_remove" messages in state and then calling human
            if (
                len([msg for msg in state["messages"] if hasattr(msg, "to_remove")])
                >= 3
            ):
                print("more than 3 repeats")
                # remove all messages (with and without "to_remove") since first "to_remove" message
                state["messages"] = state["messages"][
                    : state["messages"].index(
                        [msg for msg in state["messages"] if hasattr(msg, "to_remove")][
                            0
                        ]
                    )
                ]
                human_input = input(
                    "Please suggest AI how to introduce that change correctly:"
                )
                state.append(HumanMessage(content=human_input))

        return state

    def check_log(self, state):
        # Add logs
        logs = check_application_logs()
        log_message = HumanMessage(content="Logs:\n" + logs)

        state["messages"].append(log_message)
        return state

    # Conditional edge functions
    def after_agent_condition(self, state):
        last_message = state["messages"][-1]

        if last_message.content == bad_json_format_msg:
            return "agent"
        elif last_message.tool_call["tool"] != "final_response":
            return "tool"
        else:
            return "check_log" if log_file_path else "human"

    def after_check_log_condition(self, state):
        last_message = state["messages"][-1]

        if last_message.content.endswith("Logs are correct"):
            return "human"
        else:
            return "agent"

    # just functions
    def exchange_file_contents(self, state):
        # Remove old one
        state["messages"] = [
            msg
            for msg in state["messages"]
            if not hasattr(msg, "contains_file_contents")
        ]
        # Add new file contents
        file_contents = check_file_contents(self.files)
        file_contents_msg = HumanMessage(
            content=f"File contents:\n{file_contents}", contains_file_contents=True
        )
        state["messages"].append(file_contents_msg)
        return state

    def do_task(self, task, plan, file_contents):
        print("Executor starting its work")
        inputs = {
            "messages": [
                system_message,
                HumanMessage(content=f"Task: {task}\n\n###\n\nPlan: {plan}"),
                HumanMessage(
                    content=f"File contents: {file_contents}",
                    contains_file_contents=True,
                ),
            ]
        }
        self.executor.invoke(inputs, {"recursion_limit": 250})["messages"][-1]
