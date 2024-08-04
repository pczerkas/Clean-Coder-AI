import os
from typing import Annotated, Sequence, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain.output_parsers import XMLOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from utilities.langgraph_common_functions import (
    after_ask_human_condition,
    ask_human,
    call_model,
)
from utilities.retrying_context_manager import Retry
from utilities.util_functions import print_wrapped

load_dotenv(find_dotenv(), override=True)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.3,
# ).with_config({"run_name": "Planer"})

# llm = ChatOpenAI(model="gpt-4-vision-preview", temperature=0.3).with_config({"run_name": "Planer"})
# llm_voter = ChatAnthropic(model='claude-3-opus-20240229')
# llm = ChatOllama(model="mixtral") #, temperature=0)

# llm = ChatAnthropic(
#     model='claude-3-5-sonnet-20240620',
#     temperature=0.3,
# ).with_config({"run_name": "Planer"})

# llm = ChatOpenAI(
#     #model='deepseek-chat',
#     model='deepseek-coder',
#     openai_api_key=deepseek_api_key,
#     #openai_api_base='https://api.deepseek.com/v1',
#     openai_api_base='https://api.deepseek.com/beta',
#     temperature=0.0,
#     max_tokens=8000,
# ).with_config({"run_name": "Planer"})
# llm = ChatGroq(
#     #model="llama3-70b-8192", # too many requests (tokens)
#     #model="llama-3.1-405b-reasoning", # 404 model does not exists
#     #model="llama-3.1-70b-versatile",
#     model="Mixtral-8x7b-32768", # stupid
#     temperature=0.3,
# ).with_config({"run_name": "Planer"})
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    # convert_system_message_to_human=True,
).with_config({"run_name": "Planer"})

llm_voter = llm.with_config({"run_name": "Voter"})
llm_secretary = llm


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    voter_messages: Sequence[BaseMessage]
    secretary_messages: Sequence[BaseMessage]


system_message = SystemMessage(
    content="""
You are senior programmer. You guiding your code monkey friend about what changes need to be done in code in order
to execute given task. Think step by step and provide detailed plan about what code modifications needed to be done
to execute task. When possible, plan consistent code with other files. Your recommendations should include in details:
- Details about functions modifications - provide only functions you want to replace, without rest of the file,
- Details about movement lines and functionalities from file to file,
- Details about new file creation,
Plan should not include library installation or tests or anything else unrelated to code modifications.
At every your message, you providing proposition of all changes, not just some.

Do not rewrite full code, instead only write changes and point places where they need to be inserted.
Show with pluses (+) and minuses (-), where you want to add/remove code.
Example:
- self.id_number = str(ObjectId())
+ self.user_id = str(ObjectId())
+ self.email = email

"""
)


voter_system_message = SystemMessage(
    content="""
    Several implementation plans for a task implementation have been proposed. Carefully analyze these plans and
    determine which one accomplishes the task most effectively.
    Take in account the following criteria:
    1. The primary criterion is the effectiveness of the plan in executing the task. It is most important.
    2. A secondary criterion is simplicity. If two plans are equally good, chose one described more concise and required
    less modifications.
    3. The third criterion is consistency with existing code in other files. Prefer plan with code more similar to existing codebase.

    Respond in xml:
    ```xml
    <response>
        <evaluating>
            Neatly summarize characteristics of plan propositions. Impartially evaluate pros and cons of every of them.
        </evaluating>
        <weighting>
           Think step by step about why one proposition is better than another. Make final decision which of them is the
           best according to provided criteria.
        </weighting>
        <choice>
           Provide here nr of plan you chosen. Only the number and nothing more.
        </choice>
    </response>
    ```
    """
)

secretary_system_message = SystemMessage(
    content="""
You are secretary of lead developer. You have provided plan proposed by lead developer. Analyze the plan and find if all
proposed changes are related to provided list of project files only, or lead dev need to check other files also.

Return in:
```xml
<response>
<reasoning>
Think step by step if some additional files are needed for that plan or not.
</reasoning>
<message_to_file_researcher>
Write 'No any additional files needed.' if all the proposed plan changes are in given files; write custom message with
request to check out files in filesystem if plan assumes changes in another files than provided or lead dev wants to
ensure about something in another files.
</message_to_file_researcher>
<response>
```
"""
)


# node functions
def call_planers(state):
    messages = state["messages"]
    nr_plans = 3
    print(f"\nGenerating plan propositions...")

    def generate_plan_propositions(messages, nr_plans):
        return llm.batch(
            [messages for _ in range(nr_plans)],
            # config={'max_concurrency': 1}, # for debugging
        )

    with Retry(
        generate_plan_propositions,
        stop_max_attempt_number=3,
        wait_fixed=2000,
        before_attempts=lambda attempt_number: print(
            f"Attempting to generate plan propositions...{attempt_number}"
        ),
    ) as retry_generate_plan_propositions:
        plan_propositions_messages = retry_generate_plan_propositions(
            messages, nr_plans
        )

    for i, message in enumerate(plan_propositions_messages):
        print(f"Proposition nr {i+1}:\n\n{message.content}")

    for i, proposition in enumerate(plan_propositions_messages):
        state["voter_messages"].append(AIMessage(content="_"))
        state["voter_messages"].append(
            HumanMessage(content=f"Proposition nr {i+1}:\n\n" + proposition.content)
        )

    print("Choosing the best plan...")
    chain = llm_voter | XMLOutputParser()

    def choose_the_best_plan(state):
        return chain.invoke(
            state["voter_messages"],
            # config={'max_concurrency': 1}, # for debugging
        )

    with Retry(
        choose_the_best_plan,
        stop_max_attempt_number=3,
        wait_fixed=2000,
        before_attempts=lambda attempt_number: print(
            f"Attempting to choose the best plan...{attempt_number}"
        ),
    ) as retry_choose_the_best_plan:
        response = retry_choose_the_best_plan(state)

    choice = int(response["response"][2]["choice"])
    plan = plan_propositions_messages[choice - 1]
    state["messages"].append(plan)
    print_wrapped(f"Chosen plan no.{choice}:\n\n{plan.content}")

    '''
    print("Checking files completeness...")
    files = "['MemorialProfile.vue', 'WorkPage.vue']"   # dummy files for now
    chain = llm_secretary | XMLOutputParser()
    state["secretary_messages"].append(HumanMessage(
        content=f"""
        Plan:\n\n{plan.content}\n\n###\n\nFiles:\n\n{files}\n
    """
    ))
    secretary_response = chain.invoke(state["secretary_messages"])
    print(secretary_response)
    msg_to_file_researcher = secretary_response["response"][1]["message_to_file_researcher"]

    if msg_to_file_researcher != "No any additional files needed.":
        pass
    '''

    return state


def call_model_corrector(state):
    state, response = call_model(state, llm)
    return state


# workflow definition
researcher_workflow = StateGraph(AgentState)

researcher_workflow.add_node("planers", call_planers)
researcher_workflow.add_node("agent", call_model_corrector)
researcher_workflow.add_node("human", ask_human)
researcher_workflow.set_entry_point("planers")

researcher_workflow.add_edge("planers", "human")
researcher_workflow.add_edge("agent", "human")
researcher_workflow.add_conditional_edges("human", after_ask_human_condition)

researcher = researcher_workflow.compile()


def planning(task, message_to_programmer, files_contents):
    print("Planner starting its work")

    print(f"Message to programmer:\n{message_to_programmer}")
    print(f"Files contents:\n{files_contents}")

    message_content = f"Task: {task},\n\nMessage to programmer: {message_to_programmer},\n\nFiles:\n{files_contents}"
    human_message = HumanMessage(content=message_content)

    inputs = {
        "messages": [system_message, human_message],
        "voter_messages": [voter_system_message, human_message],
        "secretary_messages": [secretary_system_message],
    }

    def get_planner_response(inputs):
        return researcher.invoke(inputs, {"recursion_limit": 200})["messages"][-2]

    with Retry(
        get_planner_response,
        stop_max_attempt_number=3,
        wait_fixed=2000,
        before_attempts=lambda attempt_number: print(
            f"Attempting to get planner response...{attempt_number}"
        ),
    ) as retry_get_planner_response:
        planner_response = retry_get_planner_response(inputs)

    return planner_response.content
