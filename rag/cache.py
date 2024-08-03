import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import find_dotenv, load_dotenv
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.outputs import ChatGeneration

from utilities.fuzzy import get_fuzzy_directories

DESCRIPTIONS_FOR_QUESTION_DIR = (
    ".clean_coder/files_and_folders_descriptions_for_questions"
)

FILE_NAME_BEGIN = "[[["
FILE_NAME_END = "]]]"
QUESTION_BEGIN = "<<<"
QUESTION_END = ">>>"

load_dotenv(find_dotenv(), override=True)
work_dir = os.getenv("WORK_DIR")


def _normalize_text(text) -> str:
    return re.sub(r"(?:[^\w\s]|_)+", " ", text)


def _dump_generations_to_json(generations: RETURN_VAL_TYPE) -> str:
    """Dump generations to json.
    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    return json.dumps([generation.dict() for generation in generations])


def _load_generations_from_json(generations_json: str) -> RETURN_VAL_TYPE:
    """Load generations from json.
    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    try:
        results = json.loads(generations_json)

        # return [Generation(**generation_dict) for generation_dict in results]
        return [ChatGeneration(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )


class DescriptionsForQuestionCache(BaseCache):
    FUZZY_QUESTION_DIRECTORY_SCORE_TRESHOLD = 67

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        # llm_string (LLM version and settings)
        # is ommitted in key generation on purpose

        def try_read_file(
            file_name: str,
            question_text_normalized: str,
        ) -> Optional[RETURN_VAL_TYPE]:
            input_path = self._get_output_path(file_name, question_text_normalized)
            if input_path.exists():
                with open(input_path, "r", encoding="utf-8") as in_file:
                    value = in_file.read()

                    return _load_generations_from_json(value)

        file_name, question = self._get_prompt_parts(prompt)
        question_text_normalized = _normalize_text(question)
        description = try_read_file(file_name, question_text_normalized)
        if description:
            return description

        # check if question directory exists
        question_directory = self._get_output_path(question_text_normalized)
        if question_directory.exists():
            # will call update()
            return None

        fuzzy_question_directories = get_fuzzy_directories(
            question,
            parent_directory=DESCRIPTIONS_FOR_QUESTION_DIR,
        )

        if not fuzzy_question_directories:
            # will call update()
            return None

        fuzzy_question_directory_score = max(fuzzy_question_directories.values())
        if (
            fuzzy_question_directory_score
            < self.FUZZY_QUESTION_DIRECTORY_SCORE_TRESHOLD
        ):
            # will call update()
            return None

        fuzzy_question_directory = max(
            fuzzy_question_directories, key=fuzzy_question_directories.get
        )
        description = try_read_file(file_name, fuzzy_question_directory)
        if description:
            return description

        # will call update()
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        # llm_string (LLM version and settings)
        # is ommitted in key generation on purpose

        def write_file(
            file_name: str,
            question_text_normalized: str,
            return_val: RETURN_VAL_TYPE,
        ) -> None:
            output_path = self._get_output_path(file_name, question_text_normalized)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as out_file:
                value = _dump_generations_to_json(return_val)
                out_file.write(value)

        file_name, question = self._get_prompt_parts(prompt)
        question_text_normalized = _normalize_text(question)

        # check if question directory exists
        question_directory = self._get_output_path(question_text_normalized)
        if question_directory.exists():
            write_file(
                file_name,
                question_text_normalized,
                return_val,
            )

            return

        fuzzy_question_directories = get_fuzzy_directories(
            question,
            parent_directory=DESCRIPTIONS_FOR_QUESTION_DIR,
        )

        fuzzy_question_directory = None
        if fuzzy_question_directories:
            fuzzy_question_directory_score = max(fuzzy_question_directories.values())
            if (
                fuzzy_question_directory_score
                >= self.FUZZY_QUESTION_DIRECTORY_SCORE_TRESHOLD
            ):
                fuzzy_question_directory = max(
                    fuzzy_question_directories, key=fuzzy_question_directories.get
                )

        if not fuzzy_question_directory:
            # create new directory
            write_file(
                file_name,
                question_text_normalized,
                return_val,
            )

            return

        # update file in fuzzy directory
        write_file(
            file_name,
            fuzzy_question_directory,
            return_val,
        )

    def clear(self, **kwargs: Any) -> None:
        self._cache = {}

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        return self.lookup(prompt, llm_string)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        self.update(prompt, llm_string, return_val)

    async def aclear(self, **kwargs: Any) -> None:
        self.clear()

    def _find_equvalent_question_path(self, question: str) -> Path:

        return Path(work_dir + DESCRIPTIONS_FOR_QUESTION_DIR) / question

    def _get_prompt_parts(self, prompt_json: str) -> Tuple[str, str, str]:
        prompt_content = self._get_prompt_content(prompt_json)

        file_name = prompt_content.split(FILE_NAME_BEGIN, 1)[1].split(FILE_NAME_END, 1)[
            0
        ]
        question = prompt_content.split(QUESTION_BEGIN, 1)[1].split(QUESTION_END, 1)[0]

        return file_name, question

    def _get_prompt_content(self, prompt_json: str) -> str:
        prompt_data = json.loads(prompt_json)

        return prompt_data[0]["kwargs"]["content"]

    def _get_output_path(self, id: str, subdir_id: str = None) -> Path:
        def encode_name(name: str) -> str:
            return name.replace("/", "=").replace(" ", "=")

        id_encoded = encode_name(id)
        if subdir_id:
            subdir_id_encoded = encode_name(subdir_id)

            return (
                Path(work_dir + DESCRIPTIONS_FOR_QUESTION_DIR)
                / subdir_id_encoded
                / f"{id_encoded}.txt"
            )

        return Path(work_dir + DESCRIPTIONS_FOR_QUESTION_DIR) / id_encoded
