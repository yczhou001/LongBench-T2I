import json
import simplejson
from typing import Union, List
import os
def save_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())
def save_jsonl(path: str, data: Union[dict, List[dict]], append: bool = False):
    """
    Save a dictionary or a list of dictionaries to a .jsonl file.

    Args:
        path (str): Output file path.
        data (dict or List[dict]): The data to write.
        append (bool): If True, append to file; otherwise overwrite.

    Raises:
        ValueError: If data is neither a dict nor a list of dicts.
    """
    mode = 'a' if append or isinstance(data, dict) else 'w'
    with open(path, mode, encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif isinstance(data, dict):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            raise ValueError("Data must be a dict or a list of dicts")
def load_jsonl(path: str) -> List[dict]:
    """
    Load a .jsonl (JSON Lines) file and return a list of dictionaries.

    Args:
        path (str): Path to the .jsonl file.

    Returns:
        List[dict]: List of JSON objects loaded from file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, data: Union[dict, List[dict]], append: bool = False):
    """
    Save a dictionary or a list of dictionaries to a .jsonl file.

    Args:
        path (str): Output file path.
        data (dict or List[dict]): The data to write.
        append (bool): If True, append to file; otherwise overwrite.

    Raises:
        ValueError: If data is neither a dict nor a list of dicts.
    """
    mode = 'a' if append or isinstance(data, dict) else 'w'
    with open(path, mode, encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif isinstance(data, dict):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            raise ValueError("Data must be a dict or a list of dicts")


def is_valid_json(s):
    """
        Check whether a given string is a valid JSON list.

        Args:
            s (str): Input string.

        Returns:
            bool: True if the string is a valid JSON list, False otherwise.
    """
    try:
        parsed = simplejson.loads(s, strict=False)
        if isinstance(parsed, (list)):
            return True
        else:
            return False
    except ValueError:
        return False

def hash_scene(scene):
    """
        Generate a SHA-256 hash of a scene string.

        Args:
            scene (str): Input string representing a scene.

        Returns:
            str: SHA-256 hexadecimal hash of the input string.
    """
    import hashlib
    return hashlib.sha256(scene.encode('utf-8')).hexdigest()
