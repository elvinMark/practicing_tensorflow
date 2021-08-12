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

class ResBasicBLock(tf.keras.Model):
    def __init__(self,in_channel,out_channel,stride=1):
        super().__init__()
        self.straight = tf.keras.models.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=(1,1)),
            tf.keras.layers.Conv2D(out_channel,3,strides=(stride,stride),use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(padding=(1,1)),
            tf.keras.layers.Conv2D(out_channel,3,strides=(1,1),use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])

        if in_channel != out_channel or stride!=1:
            self.shortcut = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(out_channel,1,strides=(stride,stride)),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = None

        self.relu = tf.keras.layers.ReLU()

    def call(self,x):
        o = self.straight(x)
        if not self.shortcut is None:
            o = o + self.shortcut(x)
        return o

def create_resnet(nh,no):
    return tf.keras.models.Sequential([
        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(8,3,strides=(1,1)),
        ResBasicBLock(8,16,stride=2),
        ResBasicBLock(16,16),
        ResBasicBLock(16,32,stride=2),
        ResBasicBLock(32,32),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(nh,activation="relu"),
        tf.keras.layers.Dense(no,activation="softmax")
    ])

create_model_dict = {"mlp":create_mlp,"cnn":create_cnn,"resnet":create_resnet}

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
