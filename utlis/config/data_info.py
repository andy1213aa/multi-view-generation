import tensorflow as tf

training_info = {
    'save_model': {
        'logdir': '/home/aaron/Desktop/Aaron/College-level_Applied_Research/gait_log/gait_recognition'
    }
}


casia_B_train = {
    "feature": {
        "angle": tf.float32,
        "subject": tf.float32,
        "data_row": tf.string
    },

    "resolution": {
        "height": 64,
        "width": 64,
        "channel": None
    },

    "training_info": {
        "size": 73,
        "batch_size": 64,
        "shuffle": True
    }

}

OU_MVLP_triplet_train = {

    "feature": {
        "imgs": tf.string,
        "subject": tf.string,
        "angles": tf.string
        
    },

    "resolution": {
        "height": 128,
        "width": 88,
        "channel": 3,
        "angle_nums": 14,
        "k" : 4
    },

    "training_info": {
        "tfrecord_path": '/home/aaron/Desktop/Aaron/College-level_Applied_Research/tfrecord/OUMVLP_Triplet/triplet_train_4inPerson.tfrecords',
        "data_num": 50000,
        "batch_size": 16,
        "shuffle": True
    }
}
