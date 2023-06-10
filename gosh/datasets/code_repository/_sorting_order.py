import os
import re
from typing import List

from ._chunked_files import read_file_from_chunks

def get_absolute_path(file_path: str, relative_path: str) -> str:
    """Converts a relative path to an absolute path."""
    return os.path.normpath(os.path.join(os.path.dirname(file_path), relative_path))


def get_file_dependencies(path: str) -> List[str]:
    dependencies = []
    if os.path.isdir(path):
        content = read_file_from_chunks(path)
    else:
        with open(path, 'r') as file:
            content = file.read()

    matches = re.findall(r'import\s.*"(.+?)";', content)
    for dependency in matches:
        if dependency.startswith("."):
            absolute_path = get_absolute_path(path, dependency)
            dependencies.append(absolute_path)
        else:
            dependencies.append(dependency)
    return dependencies


def topological_sort(file_paths: List[str]) -> List[str]:
    graph = {
        file_path: get_file_dependencies(file_path) 
        for file_path in file_paths
    }
    sorted_files = []
    visited = set()
    # account for errors
    known_files = set(file_paths)

    def visit(file):
        if file in visited:
            return
        if file not in known_files:
            return
        visited.add(file)
        for dependency in graph[file]:
            visit(dependency)
        sorted_files.append(file)

    for file in file_paths:
        visit(file)

    return sorted_files

