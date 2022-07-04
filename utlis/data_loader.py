
import numpy as np
import tensorflow as tf



class GenericTFLoader():
    '''
    load tfrecord data.
    '''

    def __init__(self, config):
        self._config = config

    def read(self):
        raise NotImplementedError

    def parse(self):
        raise NotImplementedError

    # @classmethod
    # def generate_loader(cls, loader_subclasses, config):
    #     loader_collection = []
    #     for loader in loader_subclasses:
    #         loader_subclass()



class OU_MVLP_triplet(GenericTFLoader):

    def __init__(self, config):
        self._config = config
        self.strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    def read(self):

        
        BATCH_SIZE = self._config['training_info']['batch_size']  # per replica batch size

        # initialize tf.distribute.MirroredStrategy
        
        GLOBAL_BATCH_SIZE = self.strategy.num_replicas_in_sync * BATCH_SIZE

        print(f'Number of devices: {self.strategy.num_replicas_in_sync}')




        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_set = tf.data.TFRecordDataset(
            self._config['training_info']['tfrecord_path'])
        data_set = data_set.map(self.parse, num_parallel_calls=AUTOTUNE)
        data_set = data_set.cache()
        data_set = data_set.shuffle(
            10000, reshuffle_each_iteration=self._config['training_info']['shuffle'])
        data_batch = data_set.batch(
            GLOBAL_BATCH_SIZE, drop_remainder=True)
        data_batch = data_batch.prefetch(buffer_size=AUTOTUNE)
        data_batch_ds = self.strategy.experimental_distribute_dataset(data_batch)

        return data_batch_ds

    def parse(self, example_proto):

        features = tf.io.parse_single_example(
            example_proto,
            features={key: tf.io.FixedLenFeature(
                [], self._config['feature'][key]) for key in self._config['feature']}

        )

        imgs = features['imgs']
        subject = features['subject']
        angles = features['angles']

        imgs = tf.io.decode_raw(imgs, tf.float32)
        imgs = tf.reshape(imgs,  (self._config['resolution']['k'], 128, 88, 3))
        imgs = imgs /255.

        subject = tf.io.decode_raw(subject, tf.float32)
        subject = tf.reshape(subject, (4,))
        

        return [imgs, subject]
