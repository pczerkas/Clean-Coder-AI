import base64
import os
import re
from pathlib import Path

import playwright
from dotenv import find_dotenv, load_dotenv
from langchain.tools import StructuredTool, tool

from rag.retrieval import retrieve
from utilities.syntax_checker_functions import check_syntax
from utilities.util_functions import set_docstring

load_dotenv(find_dotenv(), override=True)
work_dir = os.getenv("WORK_DIR")
allowed_paths = list(filter(None, re.split(r"[\n,]", os.getenv("ALLOWED_PATHS"))))
read_only_paths = list(filter(None, re.split(r"[\n,]", os.getenv("READ_ONLY_PATHS"))))
blacklisted_paths = list(
    filter(None, re.split(r"[\n,]", os.getenv("BLACKLISTED_PATHS")))
)
tool_description_read_only_paths = (
    f"""This tool cannot be used in these directories (and subdirectories):
    {"\n    ".join(read_only_paths)}"""
    if read_only_paths
    else ""
)
tool_description_end = "-" * 80

WRONG_EXECUTION_WORD = "Changes have not been introduced. "

syntax_error_insert_code = """
Changes can cause next error: {error_response}. Probably you:
- Provided a wrong line number to insert code, or
- Forgot to add an indents on beginning of code.
Please analyze which place is correct to introduce the code before calling a tool.
"""
syntax_error_modify_code = """
Changes can cause next error: {error_response}. Probably you:
- Provided a wrong end or beginning line number (end code line happens more often), or
- Forgot to add an indents on beginning of code.
Think step by step which function/code block you want to change before proposing improved change.
"""


@tool
@set_docstring(
    f"""List files in directory.
    tool input:
    :param directory: Name of directory to list files in.

    Main directories:
    {'\n    '.join(allowed_paths)}
{tool_description_end}"""
)
def list_dir(directory):
    try:
        directory = "/" + directory.lstrip("/").rstrip("/")
        files = set()
        if any(directory.startswith("/" + ap.lstrip("/")) for ap in allowed_paths):
            files = os.listdir(work_dir + directory)
        elif any(
            (("/" + ap.lstrip("/")).rstrip("/") + "/").startswith(directory)
            for ap in allowed_paths
        ):
            level = (directory.rstrip("/") + "/").count("/")
            for ap in allowed_paths:
                try:
                    files.add((ap.rstrip("/").split("/"))[level - 1])
                except IndexError:
                    continue
            files = list(files)

        for f in files:
            if any(
                (directory + "/" + f).startswith("/" + bp.lstrip("/"))
                for bp in blacklisted_paths
            ):
                files.remove(f)

        return files
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Check contents of file.
    tool input:
    :param filename: Name and path of file to check.
{tool_description_end}"""
)
def see_file(filename):
    try:
        if any(filename.startswith("/" + bp.lstrip("/")) for bp in blacklisted_paths):
            return "You are not allowed to see into this file."
        with open(work_dir + filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
        formatted_lines = [f"{i+1}|{line[:-1]}\n" for i, line in enumerate(lines)]
        file_content = "".join(formatted_lines)
        file_content = filename + ":\n\n" + file_content

        return file_content
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Use that function to find files or folders in the app by text search.
    You can search for example for common styles, endpoint with user data, etc.
    Useful, when you know what do you look for, but don't know where.
    But, if it is possible, try to formulate your query in line with a main task.

    Use that function at least once BEFORE calling final response to ensure you found all appropriate files.

    tool input:
    :param query: Semantic query describing subject you looking for in one sentence. Ask for a single thing only.
{tool_description_end}"""
)
def retrieve_files_by_semantic_query(query):
    return retrieve(query)


@tool
@set_docstring(
    f"""Sees the image.
    tool input:
    :param filename: Name and path of image to check.
{tool_description_end}"""
)
def see_image(filename):
    try:
        with open(work_dir + filename, "rb") as image_file:
            img_encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return img_encoded
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Insert new piece of code into provided file. Use when new code need to be added without replacing old one.
    Proper indentation is important.
    tool input:
    :param filename: Name and path of file to change.
    :param line_number: Line number to insert new code after.
    :param code: Code to insert into the file. Without backticks around. Start it with appropriate indentation if needed.
{tool_description_end}"""
)
def insert_code(filename, line_number, code):
    try:
        with open(work_dir + filename, "r+", encoding="utf-8") as file:
            file_contents = file.readlines()
            file_contents.insert(line_number, code + "\n")
            file_contents = "".join(file_contents)
            check_syntax_response = check_syntax(file_contents, filename)
            if check_syntax_response != "Valid syntax":
                print("Wrong syntax provided, asking to correct.")
                return WRONG_EXECUTION_WORD + syntax_error_insert_code.format(
                    error_response=check_syntax_response
                )
            human_message = input(
                "Write 'ok' if you agree with agent or provide commentary: "
            )
            if human_message != "ok":
                return (
                    WRONG_EXECUTION_WORD
                    + f"Action wasn't executed because of human interruption. He said: {human_message}"
                )
            file.seek(0)
            file.truncate()
            file.write(file_contents)
        return "Code inserted."
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Replace old piece of code between start_line and end_line with new one. Proper indentation is important.
    Use that tool when you want to replace old piece of code with new one. Make smallest posible changes with this tool.
    tool input:
    :param filename: Name and path of file to change.
    :param start_line: Start line number to replace with new code. Inclusive - means start_line will be first line to change.
    :param code: New piece of code to replace old one. Without backticks around. Start it with appropriate indentation if needed.
    :param end_line: End line number to replace with new code. Inclusive - means end_line will be last line to change.

    {tool_description_read_only_paths}
{tool_description_end}"""
)
def replace_code(filename, start_line, code, end_line):
    try:
        with open(work_dir + filename, "r+", encoding="utf-8") as file:
            file_contents = file.readlines()
            file_contents[start_line - 1 : end_line] = [code + "\n"]
            file_contents = "".join(file_contents)
            check_syntax_response = check_syntax(file_contents, filename)
            if check_syntax_response != "Valid syntax":
                print(check_syntax_response)
                return WRONG_EXECUTION_WORD + syntax_error_modify_code.format(
                    error_response=check_syntax_response
                )
            human_message = input(
                "Write 'ok' if you agree with agent or provide commentary: "
            )
            if human_message != "ok":
                return (
                    WRONG_EXECUTION_WORD
                    + f"Action wasn't executed because of human interruption. He said: {human_message}"
                )
            file.seek(0)
            file.truncate()
            file.write(file_contents)
        return "Code modified."
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Create new file with provided code. Use that tool when want to insert some additional lines into code.
    tool input:
    :param filename: Name and path of file to create.
    :param code: Code to write in the file.

    {tool_description_read_only_paths}
{tool_description_end}"""
)
def create_file_with_code(filename, code):
    try:
        human_message = input(
            "Write 'ok' if you agree with agent or provide commentary: "
        )
        if human_message != "ok":
            return (
                WRONG_EXECUTION_WORD
                + f"Action wasn't executed because of human interruption. He said: {human_message}"
            )

        output_path = Path(work_dir + filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(code)
        return "File been created successfully."
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Create new directory with provided name. Use that tool when you want to create a new directory.
    tool input:
    :param path: Path of directory to create.

    {tool_description_read_only_paths}
{tool_description_end}"""
)
def create_directory(path):
    try:
        human_message = input(
            "Write 'ok' if you agree with agent or provide commentary: "
        )
        if human_message != "ok":
            return (
                WRONG_EXECUTION_WORD
                + f"Action wasn't executed because of human interruption. He said: {human_message}"
            )

        os.makedirs(work_dir + path, exist_ok=True)

        return "Directory has been created successfully."
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Rename directory with provided name. Use that tool when you want to rename directory.
    tool input:
    :param old_path: old path of directory.
    :param new_path: new path of directory.

    {tool_description_read_only_paths}
{tool_description_end}"""
)
def rename_directory(old_path, new_path):
    try:
        human_message = input(
            "Write 'ok' if you agree with agent or provide commentary: "
        )
        if human_message != "ok":
            return (
                WRONG_EXECUTION_WORD
                + f"Action wasn't executed because of human interruption. He said: {human_message}"
            )

        os.rename(work_dir + old_path, work_dir + new_path)

        return "Directory has been renamed successfully."
    except Exception as e:
        return f"{type(e).__name__}: {e}"


@tool
@set_docstring(
    f"""Ask human to provide debug actions or observations you're not available to do.
    tool input:
    :param prompt: prompt to human.
{tool_description_end}"""
)
def ask_human_tool(prompt):
    try:
        human_message = input(prompt)
        return human_message
    except Exception as e:
        return f"{type(e).__name__}: {e}"


# function under development
def make_screenshot(endpoint, login_needed, commands):
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()
    if login_needed:
        page.goto("http://localhost:5555/login")
        page.fill("#username", "uname")
        page.fill("#password", "passwd")
        page.click('.login-form button[type="submit"]')
    page.goto(f"http://localhost:5555/{endpoint}")

    for command in commands:
        action = command.get("action")
        selector = command.get("selector")
        value = command.get("value")
        if action == "fill":
            page.fill(selector, value)
        elif action == "click":
            page.click(selector)
        elif action == "hover":
            page.hover(selector)

    page.screenshot(path=work_dir + "screenshots/screenshot.png")
    browser.close()
