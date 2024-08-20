# #!/bin/bash

# # DIR="/home/ekuo/improve_logs/magic-universe-395/singleSampleTrain"
# DIR="/home/ekuo/improve_logs/magic-universe-395/train"

# cd "$DIR" || exit

# counter=0
# tar_counter=1
# file_list=""
# declare -A unique_numbers

# for file in $(ls | sort -V); do
#   first_number=$(echo "$file" | grep -o '^[0-9]\+')

#   if [[ -z "${unique_numbers[$first_number]}" ]]; then
#     unique_numbers[$first_number]=1
#     counter=$((counter + 1))
#   fi

#   file_list="$file_list $file"

#   if [[ $counter -eq 500 ]]; then
#     tar -cvf "$tar_counter.tar" $file_list
#     tar_counter=$((tar_counter + 1))
#     counter=0
#     file_list=""
#     unset unique_numbers
#     declare -A unique_numbers
#   fi
# done

# if [[ -n "$file_list" ]]; then
#   tar -cvf "$tar_counter.tar" $file_list
# fi


#!/bin/bash

# DIR="/home/ekuo/improve_logs/magic-universe-395/singleSampleTrain"
DIR="/home/ekuo/improve_logs/magic-universe-395/train"

cd "$DIR" || exit

counter=0
tar_counter=1
file_list=""
declare -A unique_numbers

for file in $(ls | sort -V); do
  first_number=$(echo "$file" | grep -o '^[0-9]\+')

  if [[ -z "${unique_numbers[$first_number]}" ]]; then
    if [[ $counter -eq 500 ]]; then
      tar -cvf "$tar_counter.tar" $file_list
      tar_counter=$((tar_counter + 1))
      counter=0
      file_list=""
      unset unique_numbers
      declare -A unique_numbers
    fi
    unique_numbers[$first_number]=1
    counter=$((counter + 1))
  fi

  file_list="$file_list $file"
done

if [[ -n "$file_list" ]]; then
  tar -cvf "$tar_counter.tar" $file_list
fi