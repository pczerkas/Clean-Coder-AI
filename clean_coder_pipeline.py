from agents.executor_agent import Executor
from agents.planner_agent import planning
from agents.researcher_agent import research_task


def run_clean_coder_pipeline(task, self_approve=False):
    files, file_contents = research_task(task)

    plan = planning(task, file_contents)

    executor = Executor(files)
    executor.do_task(task, plan, file_contents)


if __name__ == "__main__":
    task = """
    Find files related to order edit in admin
    """

    run_clean_coder_pipeline(task)
