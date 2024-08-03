import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv
from thefuzz import fuzz, process

load_dotenv(find_dotenv(), override=True)
work_dir = os.getenv("WORK_DIR")


def _normalize_text(text) -> str:
    return re.sub(r"(?:\W|_)+", " ", text)


def _is_dir(path: str) -> bool:
    return Path(work_dir + path).is_dir()


def _get_fuzzy_items(
    question: str, choices: Dict[str, str], rounds: int, cutoff_score_ratio: float
) -> Dict[str, float]:
    def score_by_ratio(min_score, max_score, ratio):
        return min_score + (max_score - min_score) * ratio

    scorers_weights = {
        fuzz.token_set_ratio: 7,  # BEST?
        fuzz.token_sort_ratio: 5,
        fuzz.partial_token_set_ratio: 2,
        fuzz.partial_token_sort_ratio: 1,
        # fuzz.QRatio: 1, # poor
        # fuzz.UQRatio: 1, # poor
        # fuzz.WRatio: 1, # nothing ...
        # fuzz.UWRatio: 1, # nothing ...
    }

    scaling_factor = rounds * sum(scorers_weights.values())

    fuzzy_items = defaultdict(float)
    for _ in range(rounds):
        for scorer, weight in scorers_weights.items():
            fuzzy_items_tuples = process.extractBests(
                _normalize_text(question),
                choices,
                # processor=default_processor,
                scorer=scorer,
                score_cutoff=0,
                limit=100,
            )

            if not fuzzy_items_tuples:
                continue

            max_score = max(fuzzy_items_tuples, key=lambda fft: fft[1])[1]
            min_score = min(fuzzy_items_tuples, key=lambda fft: fft[1])[1]
            cutoff_score = score_by_ratio(min_score, max_score, cutoff_score_ratio)
            if min_score == max_score:
                cutoff_score = 0

            for fft in fuzzy_items_tuples:
                if fft[1] <= cutoff_score:
                    continue

                fuzzy_items[fft[2]] += fft[1] * weight

    if not fuzzy_items:
        return {}

    max_item_key = max(fuzzy_items, key=fuzzy_items.get)
    min_item_key = min(fuzzy_items, key=fuzzy_items.get)
    max_score = fuzzy_items[max_item_key]
    min_score = fuzzy_items[min_item_key]

    cutoff_score = score_by_ratio(min_score, max_score, cutoff_score_ratio)
    if min_score == max_score:
        cutoff_score = 0

    fuzzy_items = {
        k: v / scaling_factor for k, v in fuzzy_items.items() if v > cutoff_score
    }

    return fuzzy_items


def get_fuzzy_files(
    question,
    parent_directories: List[str],
    filter_func: callable = None,
    rounds: int = 3,
    cutoff_score_ratio: float = 0.3,
) -> Dict[str, float]:
    paths = []
    for dir in parent_directories:
        paths.extend(
            glob(
                f"{dir}/**",
                root_dir=work_dir,
                recursive=True,
            )
        )

    def read_text(file_name: str) -> str:
        return Path(work_dir + file_name).read_text()

    def is_text_file(path: str) -> bool:
        if _is_dir(path):
            return False

        try:
            read_text(path)
            return True
        except UnicodeDecodeError:
            return False

    file_names = [
        p
        for p in paths
        if is_text_file(p)
        and (True if not filter_func else filter_func(Path(work_dir + p)))
    ]
    files_data = [
        "%s^^^%s^^^%s"
        % (
            fn,
            os.path.splitext(fn)[0].replace("/", " ").replace(".", " "),
            _normalize_text(read_text(fn)),
        )
        for fn in file_names
    ]

    choices = {k: v for k, v in zip(file_names, files_data) if v}

    return _get_fuzzy_items(question, choices, rounds, cutoff_score_ratio)


def get_fuzzy_directories(
    question,
    parent_directory: str,
    filter_func: callable = None,
    rounds: int = 3,
    cutoff_score_ratio: float = 0.3,
) -> Dict[str, float]:
    paths = glob(
        "**",
        root_dir=work_dir + parent_directory,
    )

    directory_names = [
        p
        for p in paths
        if _is_dir(parent_directory + "/" + p)
        and (
            True
            if not filter_func
            else filter_func(Path(work_dir + parent_directory + "/" + p))
        )
    ]
    directories_data = [
        "%s^^^%s"
        % (
            dn,
            _normalize_text(dn),
        )
        for dn in directory_names
    ]

    choices = {k: v for k, v in zip(directory_names, directories_data) if v}

    return _get_fuzzy_items(question, choices, rounds, cutoff_score_ratio)
