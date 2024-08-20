import os

def find_missing_files(expected_count, directory):
    expected_files = set(range(expected_count))
    actual_files = set()

    for filename in os.listdir(directory):
        if filename.endswith('.tfrecord'):
            file_number = int(filename.split('.')[0])
            actual_files.add(file_number)

    missing_files = expected_files - actual_files
    return sorted(missing_files)

# Adjust the parameters according to your directory and expected file count
directory = '/home/ekuo/improve_logs/magic-universe-395/tfrecords'
expected_count = 7147  # Since files are numbered 0 to 7146

missing_files = find_missing_files(expected_count, directory)

if missing_files:
    print("Missing file numbers:", missing_files)
else:
    print("No files are missing.")
