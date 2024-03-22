import tensorflow as tf
import glob

def count_records(tfrecord_file):
    # Create a TFRecordDataset to read the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Initialize a counter variable
    record_count = 0

    # Iterate through the dataset and count the records, handling errors
    for raw_record in dataset:
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            record_count += 1
        except tf.errors.DataLossError as e:
            print(f"Warning: Corrupted record in file {tfrecord_file}: {e}")

    return record_count


base_path = 'droid/1.0.0/r2d2_faceblur-train.tfrecord-'
filename_pattern = '{:05d}-of-02048'

# Calculate the number of records in each TFRecord file
for i in range(2048):
    tfrecord_file = base_path + filename_pattern.format(i)
    records_per_file = count_records(tfrecord_file)
    print(filename_pattern.format(i), ":", records_per_file)


