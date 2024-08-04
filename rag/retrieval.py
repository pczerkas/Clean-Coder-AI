# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
import sys

import pysqlite3

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import re
from glob import glob
from pathlib import Path

import chromadb
import cohere

# from chromadb.utils import embedding_functions
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI

from rag.cache import (
    FILE_NAME_BEGIN,
    FILE_NAME_END,
    QUESTION_BEGIN,
    QUESTION_END,
    DescriptionsForQuestionCache,
)
from rag.write_descriptions import (
    get_output_path,
    is_code_file,
    normalize_text,
    prepare_collection_document,
    upload_descriptions_to_vdb,
    write_descriptions,
)
from utilities.fuzzy import get_fuzzy_files
from utilities.retrying_context_manager import Retry

LARGE_FILE_DESCRIPTION_CHARS_LIMIT = 10000
DESCRIPTIONS_PER_QUESTION_DIR = (
    ".clean_coder/files_and_folders_descriptions_for_questions"
)

load_dotenv(find_dotenv(), override=True)
work_dir = os.getenv("WORK_DIR")
allowed_paths = list(filter(None, re.split(r"[\n,]", os.getenv("ALLOWED_PATHS"))))
read_only_paths = list(filter(None, re.split(r"[\n,]", os.getenv("READ_ONLY_PATHS"))))
blacklisted_paths = list(
    filter(None, re.split(r"[\n,]", os.getenv("BLACKLISTED_PATHS")))
)
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(cohere_key)

# llm = ChatAnthropic(
#     model='claude-3-5-sonnet-20240620',
#     cache=DescriptionsForQuestionCache(),
# )
# llm = ChatOpenAI(
#     #model='deepseek-chat',
#     model='deepseek-coder',
#     openai_api_key=deepseek_api_key,
#     #openai_api_base='https://api.deepseek.com/v1',
#     openai_api_base='https://api.deepseek.com/beta',
#     temperature=0.0,
#     max_tokens=8000,
#     cache=DescriptionsForQuestionCache(),
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    # convert_system_message_to_human=True,
    cache=DescriptionsForQuestionCache(),
)

if cohere_key:
    chroma_client = chromadb.PersistentClient(
        path=os.getenv("WORK_DIR") + ".clean_coder/chroma_base"
    )
    collection_name = f"clean_coder_{Path(work_dir).name}_file_descriptions"
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            # embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(),
        )
        vdb_availabe = True
    except ValueError:
        print(
            "Vector database does not exist. Please create it by running rag/write_descriptions.py"
        )
        vdb_availabe = False

    cohere_client = cohere.Client(cohere_key)


def _handle_large_file_description_for_question(file_description, question):
    if len(file_description) > LARGE_FILE_DESCRIPTION_CHARS_LIMIT:
        # select 1st line and add convincing comment
        file_description = (
            file_description.split("\n", 1)[0]
            + f'\nThis file is highly relevant to the question: "{question}"'
        )

    return file_description


def _get_files_descriptions_for_question(
    files_names, files_descriptions, question
) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Given file:
{FILE_NAME_BEGIN}{file_name}{FILE_NAME_END}
and its lenghty description:
'''{file_description}'''

provide what file is responsible for and at most 3 key facts about this file with regard to the question:
{QUESTION_BEGIN}{question}{QUESTION_END}. Without any comments from your side.

Facts should be labelled with letters 'a', 'b', 'c' (etc).
Warning: if there are no relevant facts about the file, or the file
is not related to the question, just return an empty string. It is important.
"""
    )

    chain = prompt | llm | StrOutputParser()

    return chain.batch(
        [
            dict(
                (k, v)
                for k, v in zip(
                    [
                        "question",
                        "file_name",
                        "file_description",
                        "FILE_NAME_BEGIN",
                        "FILE_NAME_END",
                        "QUESTION_BEGIN",
                        "QUESTION_END",
                    ],
                    [
                        question,
                        files_name,
                        _handle_large_file_description_for_question(
                            file_description, question
                        ),
                        FILE_NAME_BEGIN,
                        FILE_NAME_END,
                        QUESTION_BEGIN,
                        QUESTION_END,
                    ],
                )
            )
            for files_name, file_description in zip(files_names, files_descriptions)
        ],
        # config={'max_concurrency': 1}, # for debugging
    )


def _filter_func(file_path: Path) -> bool:
    if not is_code_file(file_path):
        return False

    def file_name_in_blacklist(file_name_relative) -> bool:
        return any(file_name_relative.startswith(b) for b in blacklisted_paths)

    return not file_name_in_blacklist(
        file_path.relative_to(work_dir).as_posix(),
    )


def retrieve(question):
    DIRECTORY_FILES_COUNT_LIMIT = 30
    MIN_DESCRIPTION_FOR_QUESTION_LENGTH = 20

    fuzzy_files = get_fuzzy_files(
        question,
        allowed_paths,
        filter_func=_filter_func,
    )

    for i, fuzzy_file in enumerate(fuzzy_files):
        item_number = i + 1
        print(f"{item_number}. fuzzy file: {fuzzy_file}")

    directories = [re.sub("/etc$", "", os.path.dirname(f)) for f in fuzzy_files]
    directories = list(set(directories))

    large_directories = []

    def get_directory_files_count(directory):
        # skip check if directory is parent drectory of a large directory
        if any(dd.startswith(directory) for dd in large_directories):
            return DIRECTORY_FILES_COUNT_LIMIT + 1

        count = len(glob(work_dir + directory + "/**", recursive=True))
        if count > DIRECTORY_FILES_COUNT_LIMIT:
            large_directories.append(directory)

        return count

    # sort directories by name length descending
    directories = sorted(directories, key=len, reverse=True)

    # exclude large directories
    directories = [
        d
        for d in directories
        if get_directory_files_count(d) <= DIRECTORY_FILES_COUNT_LIMIT
    ]

    file_paths = write_descriptions(directories=directories)
    if file_paths:
        upload_descriptions_to_vdb(file_paths)

    # embeding_func = embedding_functions.DefaultEmbeddingFunction()
    # embeding_func = embedding_functions.SentenceTransformerEmbeddingFunction()
    # query_embeddings = embeding_func([normalize_text(question)])

    retrieval = collection.query(
        query_texts=[normalize_text(question)],
        # query_embeddings=query_embeddings,
        n_results=100,
    )
    # print('retrieval: ', retrieval)

    retrieval_ids = retrieval["ids"][0]
    retrieval_documents = retrieval["documents"][0]

    for i, retrieval_file_name in enumerate(retrieval_ids):
        item_number = i + 1
        print(f"{item_number}. retrieval file: {retrieval_file_name}")

    for fuzzy_file in fuzzy_files:
        if fuzzy_file not in retrieval_ids:
            file_paths = write_descriptions(files=[fuzzy_file])
            if file_paths:
                upload_descriptions_to_vdb(file_paths)

                file_path = get_output_path(Path(work_dir + fuzzy_file))
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                retrieval_ids.append(fuzzy_file)
                retrieval_documents.append(
                    prepare_collection_document(fuzzy_file, content)
                )

    fuzzy_files_in_retrieval_indexes = [
        idx
        for id, idx in zip(
            retrieval_ids, [j for j, _ in enumerate(retrieval_documents)]
        )
        if id in fuzzy_files
    ]

    for i, retrieval_file_name in enumerate(retrieval_ids):
        item_number = i + 1
        print(
            f"{item_number}. retrieval file (after merge with fuzzy files): {retrieval_file_name}"
        )

    def get_reranked_docs(question):
        return cohere_client.rerank(
            # query=question,
            query=normalize_text(question),
            # documents=retrieval_documents,
            # documents=[normalize_text(doc) for doc in retrieval_documents],
            documents=[
                _handle_large_file_description_for_question(doc, question)
                for doc in retrieval_documents
            ],
            # top_n=4,
            top_n=35,
            model="rerank-english-v3.0",
            # return_documents=True,
        )

    with Retry(
        get_reranked_docs,
        stop_max_attempt_number=3,
        wait_fixed=2000,
        before_attempts=lambda attempt_number: print(
            f"Attempting to rerank documents...{attempt_number}"
        ),
    ) as retry_get_reranked_docs:
        reranked_docs = retry_get_reranked_docs(question)
    # print("reranked_docs: ", reranked_docs)

    reranked_indexes = [result.index for result in reranked_docs.results]

    # always add fuzzy files in retrieval
    for idx in fuzzy_files_in_retrieval_indexes:
        if idx not in reranked_indexes:
            reranked_indexes.append(idx)

    for i, index in enumerate(reranked_indexes):
        item_number = i + 1
        print(f"{item_number}. reranked file: ", retrieval_ids[index])

    files_names = [retrieval_ids[index] for index in reranked_indexes]
    files_descriptions = [
        # normalize_text(retrieval_documents[idx])
        retrieval_documents[idx]
        for idx in reranked_indexes
    ]

    with Retry(
        _get_files_descriptions_for_question,
        stop_max_attempt_number=3,
        wait_fixed=2000,
        before_attempts=lambda attempt_number: print(
            f"Attempting to get files descriptions for question...{attempt_number}"
        ),
    ) as retry_get_files_descriptions_for_question:
        files_descriptions_for_question = retry_get_files_descriptions_for_question(
            files_names, files_descriptions, question
        )
    # print(*files_descriptions_for_question, sep='\n\n')

    response = ""
    item_number = 1
    for filename, description_for_question in zip(
        files_names, files_descriptions_for_question
    ):
        if (
            not description_for_question
            or len(description_for_question) < MIN_DESCRIPTION_FOR_QUESTION_LENGTH
            or any(
                p in description_for_question.lower()
                for p in ["not related", "does not directly relate", "does not relate"]
            )
        ):
            continue

        is_read_only = any(
            ("/" + filename.lstrip("/")).rstrip("/").startswith("/" + rop.lstrip("/"))
            for rop in read_only_paths
        )
        filename_line = (
            f"{item_number}. File {filename}"
            + (
                " (this file is read-only - you cannot modify it in any way!)"
                if is_read_only
                else ""
            )
            + ":"
        )

        response += f"{filename_line}\n{description_for_question}\n\n"
        item_number += 1

    response += "\n\nRemember to see files (use tool `see_file`) before adding to final response!"
    print(response)

    return response


if __name__ == "__main__":
    results = retrieve("Find files that reference or import some module")
    print("\n\n")
    print("results: ", results)
