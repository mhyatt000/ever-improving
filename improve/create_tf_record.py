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

def create_tfrecords(data_directory, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for filename in os.listdir(data_directory):
            if filename.endswith('.mp4') or filename.endswith('.pt'):
                sample_number = int(filename.split('.')[0])

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
                writer.write(example)

data_directory = '/home/ekuo/improve_logs/magic-universe-395/train'
# output_file = '/home/ekuo/improve_logs/magic-universe-395/magic_universe.tfrecords'
# data_directory = '/home/ekuo/improve_logs/magic-universe-395/singleSampleTrain'
output_file = '/home/ekuo/improve_logs/magic-universe-395/single.tfrecords'
create_tfrecords(data_directory, output_file)
