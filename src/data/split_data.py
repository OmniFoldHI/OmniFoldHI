import numpy as np
import argparse

def get_header (filename):

    header = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                header.append(line)
            else:
                return header

# Get data file to split in two
parser = argparse.ArgumentParser(description='Script to slit a data file into two')
parser.add_argument('-f', '--data_file', type=str, help='File to slit in two.')
args = parser.parse_args()

# Chosen file
data_file = args.data_file

# Get header
data_header = get_header(data_file)

# Get data
data = np.loadtxt(data_file)

# Create new file with fisrt half of data
data_file_A = f"{data_file.split('.')[0]}_A.{data_file.split('.')[-1]}"
data_file_A = f"data/closure_test/{data_file_A.split('/')[-1]}"
with open(data_file_A, 'w') as file_A:

    # Add header
    file_A.writelines(data_header)

    # Add data
    np.savetxt(file_A, data[:len(data)//2], fmt='%s')

# Create new file with second half of data
data_file_B = f"{data_file.split('.')[0]}_B.{data_file.split('.')[-1]}"
data_file_B = f"data/closure_test/{data_file_B.split('/')[-1]}"
with open(data_file_B, 'w') as file_B:

    # Add header
    file_B.writelines(data_header)

    # Add data
    np.savetxt(file_B, data[len(data)//2:], fmt='%s')

# Print created files
print(f'Created files: {data_file_A}; {data_file_B}')