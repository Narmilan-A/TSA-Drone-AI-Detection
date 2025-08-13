import os

def extract_tree_structure(root_dir, output_file):
    def write_tree(current_dir, f, indent_level=0):
        indent = "│   " * indent_level
        folder_name = os.path.basename(current_dir)
        f.write(f"{indent}├── {folder_name}/\n")
        
        try:
            items = sorted(os.listdir(current_dir))
        except PermissionError:
            f.write(f"{indent}│   [Permission Denied]\n")
            return

        for item in items:
            path = os.path.join(current_dir, item)
            if os.path.isdir(path):
                write_tree(path, f, indent_level + 1)
            else:
                f.write(f"{indent}│   └── {item}\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        write_tree(root_dir, f)

# Example usage
extract_tree_structure('N:/uow/tasks/SAEF-UAS Project', 'N:/uow/tasks/SAEF-UAS Project/metadata_structure.txt')


