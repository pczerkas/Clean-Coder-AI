# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import re
from pathlib import Path
from typing import List

import chromadb

# from chromadb.utils import embedding_functions
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI

from utilities.retrying_context_manager import Retry

DESCRIPTIONS_DIR = ".clean_coder/files_and_folders_descriptions"

load_dotenv(find_dotenv(), override=True)
work_dir = os.getenv("WORK_DIR")
allowed_paths = list(filter(None, re.split(r'[\n,]', os.getenv("ALLOWED_PATHS"))))
blacklisted_paths = list(filter(None, re.split(r'[\n,]', os.getenv("BLACKLISTED_PATHS"))))
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# llm = ChatAnthropic(
#     model='claude-3-5-sonnet-20240620',
# )
# llm = ChatOpenAI(
#     #model='deepseek-chat',
#     model='deepseek-coder',
#     openai_api_key=deepseek_api_key,
#     #openai_api_base='https://api.deepseek.com/v1',
#     openai_api_base='https://api.deepseek.com/beta',
#     temperature=0.0,
#     max_tokens=8000
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    # convert_system_message_to_human=True,
)


# read file content. place name of file in the top
def _get_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return file_path.name + "\n" + content


def _get_file_name_from_output_path(file_path: Path):
    return file_path.name.replace("=", "/").removesuffix(".txt")


def _file_path_has_description(file_path: Path) -> bool:
    return os.path.exists(get_output_path(file_path))


def _filter_func(file_path: Path) -> bool:
    if not is_code_file(file_path):
        return False

    def file_name_in_blacklist(file_name_relative) -> bool:
        return any(file_name_relative.startswith(b) for b in blacklisted_paths)

    return not file_name_in_blacklist(
        file_path.relative_to(work_dir).as_posix(),
    )


def is_code_file(file_path: Path) -> bool:
    # List of common code file extensions
    code_extensions = {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".vue",
        ".py",
        ".rb",
        ".php",
        ".java",
        ".c",
        ".cpp",
        ".cs",
        ".go",
        ".swift",
        ".kt",
        ".rs",
        ".htm",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".xml",
        ".json",
        # '.csv' skipped - large files
    }

    return file_path.suffix.lower() in code_extensions


def get_output_path(file_path: Path):
    file_name_relative_encoded = (
        file_path.relative_to(work_dir).as_posix().replace("/", "=")
    )

    return Path(work_dir + DESCRIPTIONS_DIR) / f"{file_name_relative_encoded}.txt"


def normalize_text(text) -> str:
    return re.sub(r"(?:[^\w\s]|_)+", " ", text)


def write_descriptions(
    directories: List[str] = [],
    files: List[str] = [],
) -> List[Path]:
    all_files_paths = []

    for dir in directories:
        for root, _, _files in os.walk(work_dir + dir):
            for file in _files:
                file_path = Path(root) / file
                if _filter_func(file_path):
                    all_files_paths.append(file_path)

    for file in files:
        file_path = Path(work_dir + file)
        if _filter_func(file_path):
            all_files_paths.append(file_path)

    all_files_paths = [
        fp for fp in all_files_paths if not _file_path_has_description(fp)
    ]

    if not all_files_paths:
        return []

    if len(all_files_paths) > 250:
        raise Exception("Too many files to describe: " + str(len(all_files_paths)))

    prompt = ChatPromptTemplate.from_template(
        # """Describe the following code in 4 sentences or less, focusing only on important information from integration point of view.
        # Write what file is responsible for.\n\n'''\n{code}'''
        # """
        """Describe the following code, focusing only on important information from
integration point of view. Write what file is responsible for.
Mention names of all imported modules (even models and exception classes).
Mention names of all functions and classes.
\n\n'''\n{code}'''
"""
    )
    chain = prompt | llm | StrOutputParser()

    def get_descriptions(files_paths):
        return chain.batch(
            [_get_content(fp) for fp in files_paths],
            # config={'max_concurrency': 1}, # for debugging
        )

    # iterate over all files, take 8 files at once
    batch_size = 8
    output_paths = []
    for i in range(0, len(all_files_paths), batch_size):
        files_paths_batch = all_files_paths[i : i + batch_size]
        undescribed_files_paths_batch = []
        for file_path in files_paths_batch:
            output_path = get_output_path(file_path)
            if not os.path.exists(output_path):
                undescribed_files_paths_batch.append(file_path)

        if not undescribed_files_paths_batch:
            continue

        with Retry(
            get_descriptions, stop_max_attempt_number=3, wait_fixed=2000
        ) as retry_get_descriptions:
            descriptions = retry_get_descriptions(undescribed_files_paths_batch)
        # print(*descriptions, sep='\n\n')

        for file_path, description in zip(undescribed_files_paths_batch, descriptions):
            output_path = get_output_path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(description)

            output_paths.append(output_path)

    return output_paths


def prepare_collection_document(file_path_id, content) -> str:
    return "%s\n%s" % (
        normalize_text(file_path_id),
        normalize_text(content),
    )


def upload_descriptions_to_vdb(limit_file_paths: List[Path] = None) -> None:
    chroma_client = chromadb.PersistentClient(
        path=os.getenv("WORK_DIR") + ".clean_coder/chroma_base"
    )
    collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        # embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
        # embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(),
    )

    # read files and upload to base
    for root, _, files in os.walk(work_dir + DESCRIPTIONS_DIR):
        for file in files:
            file_path = Path(root) / file
            file_path_id = _get_file_name_from_output_path(file_path)
            if limit_file_paths is not None:
                if not any(
                    file_path_id == _get_file_name_from_output_path(p)
                    for p in limit_file_paths
                ):
                    continue

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            collection.upsert(
                documents=[prepare_collection_document(file_path_id, content)],
                ids=[file_path_id],
            )


if __name__ == "__main__":
    force_upload = "--force-upload" in sys.argv
    file_paths = write_descriptions(directories=allowed_paths)
    if force_upload:
        upload_descriptions_to_vdb()
    elif file_paths:
        upload_descriptions_to_vdb(file_paths)
