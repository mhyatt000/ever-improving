import math
import os
import tensorflow as tf

def serialize_example(sample_number, obs, state, video, next_obs):
    feature = {
        'sample_number': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample_number])),
        'obs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[obs])),
        'state': tf.train.Feature(bytes_list=tf.train.BytesList(value=[state])),
        'video': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video])),
        'next_obs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[next_obs])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecords(data_directory, output_directory, samples_per_file):
    os.makedirs(output_directory, exist_ok=True)

    processed_samples = set()
    file_index = 0
    current_sample_count = 0
    writer = None

    for filename in os.listdir(data_directory):
        if filename.endswith('.mp4') or filename.endswith('.pt'):
            sample_number = int(filename.split('.')[0])

            if sample_number in processed_samples:
                continue

            obs_path = os.path.join(data_directory, f'{sample_number}.obs.mp4')
            state_path = os.path.join(data_directory, f'{sample_number}.pt')
            video_path = os.path.join(data_directory, f'{sample_number}.video.mp4')
            next_obs_path = os.path.join(data_directory, f'{sample_number}.next_obs.mp4')

            with open(obs_path, 'rb') as f:
                obs_data = f.read()

            with open(state_path, 'rb') as f:
                state_data = f.read()

            with open(video_path, 'rb') as f:
                video_data = f.read()

            with open(next_obs_path, 'rb') as f:
                next_obs_data = f.read()

            example = serialize_example(sample_number, obs_data, state_data, video_data, next_obs_data)

            if current_sample_count == 0:
                output_file = os.path.join(output_directory, f'{file_index}.tfrecord')
                writer = tf.io.TFRecordWriter(output_file)

            writer.write(example)
            processed_samples.add(sample_number)
            current_sample_count += 1

            if current_sample_count >= samples_per_file:
                # Close the current TFRecord file and reset counters
                writer.close()
                file_index += 1
                current_sample_count = 0
    if writer:
        writer.close()

data_directory = '/home/ekuo/improve_logs/magic-universe-395/train'
output_directory = '/home/ekuo/improve_logs/magic-universe-395/tfrecords'
samples_per_file = int(math.ceil(7146 / 8))
create_tfrecords(data_directory, output_directory, samples_per_file)
