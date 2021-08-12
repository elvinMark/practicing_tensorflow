import tensorflow as tf

def create_mlp(nh,no):
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(nh,activation='relu'),
        tf.keras.layers.Dense(no,activation='softmax')
    ])

def create_cnn(nh,no):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8,3,strides=(2,2),use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(16,3,strides=(2,2),use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(nh,activation='relu'),
        tf.keras.layers.Dense(no,activation='softmax')
    ])

create_model_dict = {"mlp":create_mlp,"cnn":create_cnn}

model_dict = {
    "mnist": {
        "mlp": {
            "nh" : 256,
            "no" : 10
        },
        "cnn": {
            "nh" : 128,
            "no" : 10
        },
        "resnet": {
            "nh" : 64,
            "no" : 10
        }
    },
    "kmnist": {
        "mlp": {
            "nh" : 256,
            "no" : 10
        },
        "cnn": {
            "nh" : 128,
            "no" : 10
        },
        "resnet": {
            "nh" : 64,
            "no" : 10
        }
    },
    "cifar10": {
        "mlp": {
            "nh" : 256,
            "no" : 10
        },
        "cnn": {
            "nh" : 128,
            "no" : 10
        },
        "resnet": {
            "nh" : 64,
            "no" : 10
        }
    },
    "cifar100": {
        "mlp": {
            "nh" : 256,
            "no" : 100
        },
        "cnn": {
            "nh" : 128,
            "no" : 100
        },
        "resnet": {
            "nh" : 256,
            "no" : 100
        }
    }
}

def create_model(args):
    return create_model_dict[args.model](**model_dict[args.dataset][args.model])
