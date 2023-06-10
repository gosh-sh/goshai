from torch.utils.data import Dataset
import tempfile
from typing import Callable
import git
import os

from ._chunked_files import read_file_from_chunks
from ._sorting_order import topological_sort

FILE_START_TOKEN = "<filename>"

SPECIAL_FILE_NAMES = ["SPEC.md", "CTO.md", "ARCH.md", "TASKS.md"]


def _file_fits_extentions_filter(file_path: str, extentions: "list[str] | None") -> bool:
    _, extention = os.path.splitext(file_path)
    return extention in extentions

def _is_sol_code(file_path: str) -> bool:
    return _file_fits_extentions_filter(file_path, [".sol", ".tsol"])

def _slit_files_by_type(path: str, extentions: "list[str] | None") -> str:
    sol_files_list = []
    special_files = []
    files_list = []

    for root, dirs, files in os.walk(path):
        if root.endswith(".tsol"):
            sol_files_list.append(root)
        else:
            for file in files:
                file_path = os.path.join(root, file)

                if file in SPECIAL_FILE_NAMES:
                    special_files.append(file_path)
                elif not os.path.islink(file_path) and os.path.isfile(file_path):
                    if extentions is None or _file_fits_extentions_filter(file_path, extentions):
                        files_list.append(file_path)
                        if _is_sol_code(file_path):
                            sol_files_list.append(file_path)
    return special_files, files_list, sol_files_list

def _load_chunked_sources_from_dir(path: str, extentions: "list[str] | None") -> str:
    special_files, files_list, sol_files_list = _slit_files_by_type(path, extentions)

    def order_special_files(files_paths, order: list[str]):
        order_mapping = { name: i for i, name in enumerate(order) }
        return sorted(files_paths, key=lambda x: order_mapping.get(os.path.split(x)[-1]))

    special_files = order_special_files(special_files, SPECIAL_FILE_NAMES)

    sol_files_list = topological_sort(sol_files_list)

    file_contents = ""

    for file_path in special_files + files_list + sol_files_list:
        try:
            if _is_sol_code(file_path):
                sol_dir, sol_filename = os.path.split(file_path)
                if sol_dir.endswith(".tsol"):
                    sol_file_content = read_file_from_chunks(sol_dir)
                    file_contents += f"{FILE_START_TOKEN}{sol_dir}\n{sol_file_content}"
                    continue

            with open(file_path, "r", errors='ignore') as f:
                content = f.read()
            file_contents += f"{FILE_START_TOKEN}{file_path}\n{content}"
        except OSError:
            pass

    return file_contents
                
def _load_sources_from_dir(path: str, extentions: "list[str] | None") -> str:
    file_contents = ""
    files_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path) and os.path.isfile(file_path):
                if extentions is None or _file_fits_extentions_filter(file_path, extentions):
                    files_list.append(file_path)

    files_list = topological_sort(files_list) 
    for file_path in files_list:
        try:
            with open(file_path, "r", errors='ignore') as f:
                content = f.read()
            file_contents += f"{FILE_START_TOKEN}{file_path}\n{content}"
        except OSError:
            pass
    
    return file_contents


def _load_sources_from_git(address: str, extentions: "list[str] | None") -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        git.Repo.clone_from(address, temp_dir)
        return _load_chunked_sources_from_dir(temp_dir, extentions), address


def _create_loader_from_dir_prefixed(prefix):
    prefix_len = len(prefix)

    def loader(path, extentions: "list[str] | None"):
        assert(path.startswith(prefix))
        address = path[prefix_len:]
        return _load_chunked_sources_from_dir(address, extentions), address

    return loader


class CodeRepository(Dataset):
    # Matches protocol prefix to it's loader
    LOADERS = {
        "git@github.com:": _load_sources_from_git,
        "github.com:": _load_sources_from_git,
        "file:": _create_loader_from_dir_prefixed("file:")
    }
 
    def __init__(self):
        self._repos = []
        self._extentions = None

    def is_valid(self, path: str) -> bool:
        return CodeRepository._find_matching_loader(path) is not None
    
    def add_path(self, path: str):
        # Assumption: 
        # same path can be added multiple times intentionally
        assert self.is_valid(path), (
            "Path provided is not valid or protocol is not supported,"
            + " therefore it can not be loaded."
        )
        self._repos.append(path)
        return self

    def set_extentions(self, extentions: "list[str] | None"):
        self._extentions = extentions

    def __len__(self):
        return len(self._repos)
    
    def __getitem__(self, idx):
        address = self._repos[idx]
        return CodeRepository._load_sources(address, self._extentions)

    @staticmethod
    def _find_matching_loader(path) -> Callable[[str], str]:
        for protocol_prefix in CodeRepository.LOADERS.keys():
            if path.startswith(protocol_prefix):
                return CodeRepository.LOADERS[protocol_prefix]
        return None

    @staticmethod
    def _load_sources(address: str, extentions: "list[str] | None") -> str:
        load_fn = CodeRepository._find_matching_loader(address)
        return load_fn(address, extentions) 

