import os

def read_file_from_chunks(dir_path: str):
    def get_files_in_directory(directory):
        files = []
        for root, _, filenames in os.walk(directory):
            if root == directory:
                files.extend(filenames)
        return files

    def order_files_by_number(file_names):
        def extract_number(file_name):
            return int(''.join(filter(str.isdigit, file_name)))

        return sorted(file_names, key=extract_number)

    files = get_files_in_directory(dir_path)
    chunk_files = [file for file in files if file.startswith("CHUNK")]
    chunk_files = order_files_by_number(chunk_files)

    file_content = ""
    for chunk_file in chunk_files:

        with open(os.path.join(dir_path, chunk_file), "r", errors='ignore') as f:
            content = f.read()
            file_content += content
    return file_content
