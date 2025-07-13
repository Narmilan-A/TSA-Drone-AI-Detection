#---------------------------------------------------------------------------------#
# File and folder structure
import os

# Define the root folder path
root_folder_path = r'F:/scc_final_submission'

# Function to retrieve folder structure and export to txt file with hierarchy
def export_folder_structure_with_hierarchy(root_path, output_file):
    with open(output_file, 'w') as f:
        for folder_path, subfolders, file_names in os.walk(root_path):
            # Calculate the depth of the current folder
            depth = folder_path.count(os.sep) - root_path.count(os.sep)

            # Create a representation of the current folder's hierarchy
            folder_hierarchy = '|   ' * depth + '|-- '

            # Write folder names and file names with the hierarchy representation
            f.write(f'{folder_hierarchy}Folder: {os.path.basename(folder_path)}\n')
            for file_name in file_names:
                f.write(f'{folder_hierarchy}File: {file_name}\n')

# Specify the output file path (where you want to export the structure)
output_file_path = 'F:/scc_final_submission/folder_and_file_structure.txt'

# Call the function to export the folder structure with hierarchy
export_folder_structure_with_hierarchy(root_folder_path, output_file_path)

print(f'Folder structure with hierarchy exported to {output_file_path}')
#---------------------------------------------------------------------------------#
# Folder structure
import os

# Define the root folder path
root_folder_path = r'F:/scc_final_submission'

# Function to retrieve folder structure and export to a more visually appealing txt file with hierarchy
def export_folder_structure_with_hierarchy(root_path, output_file):
    with open(output_file, 'w') as f:
        for folder_path, subfolders, _ in os.walk(root_path):
            # Calculate the depth of the current folder3
            depth = folder_path.count(os.sep) - root_path.count(os.sep)

            # Create a visual block representation for folders, bolding the main folders
            if depth == 0:
                folder_structure = f'|-- Folder: {os.path.basename(folder_path)}'
            else:
                folder_structure = f'|   ' * (depth - 1) + f'|------: {os.path.basename(folder_path)}'

            f.write(f'{folder_structure}\n')

# Specify the output file path (where you want to export the structure)
output_file_path = 'F:/scc_final_submission/folder_structure.txt'

# Call the function to export the folder structure with hierarchy
export_folder_structure_with_hierarchy(root_folder_path, output_file_path)

print(f'Folder structure with hierarchy exported to {output_file_path}')
#---------------------------------------------------------------------------------#







