from agents.executor_agent import Executor
from agents.planner_agent import planning
from agents.researcher_agent import research_task


def run_clean_coder_pipeline(task, self_approve=False):
    message_to_programmer, files, files_contents = research_task(task)

    plan = planning(task, message_to_programmer, files_contents)

    executor = Executor(files)
    executor.do_task(task, plan, files_contents)


if __name__ == "__main__":
    # task = """
    # Find files related to order edit in admin
    # """
    task = """
    In code using with Retry() context manager I want to add handling of 'before_attempts'
    parameter - this should be a function that prints attempt number before each attempt.
    """

    run_clean_coder_pipeline(task)
