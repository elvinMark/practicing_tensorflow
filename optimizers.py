import tensorflow as tf

def create_optimizer(args):
    if args.optim == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=args.lr)
    elif args.optim == "adam":
        return tf.keras.optimizers.Adam(learning_rate=args.lr)
    else:
        return None

