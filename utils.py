import tensorflow as tf
import numpy as np

def train_one_step(model,ds,crit,optim,train_):
    train_loss = 0
    total = 0
    correct = 0
    
    for x,y in ds:
        loss, o = train_(x,y)
        train_loss += loss
        correct += np.sum(np.argmax(o,axis=1) == y)
        total += len(y)
    return train_loss, 100 * correct / total

def validate(model,ds,crit,validate_):
    test_loss = 0
    total = 0
    correct = 0
    for x,y in ds:
        loss, o = validate_(x,y)
        test_loss += loss
        total += len(y)
        correct += np.sum(np.argmax(o,axis=1)==y)
    return test_loss, 100 * correct / total

def train(model,train_ds, test_ds, crit, optim, sched, epochs):
    @tf.function
    def train_(x,y):
        with tf.GradientTape() as tape:
            o = model(x,training=True)
            loss = crit(y,o)
        gradients = tape.gradient(loss,model.trainable_variables)
        optim.apply_gradients(zip(gradients,model.trainable_variables))
        return loss, o

    @tf.function
    def validate_(x,y):
        o = model(x,training=False)
        loss = crit(y,o)
        return loss, o
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_step(model,train_ds,crit, optim,train_)
        test_loss, test_acc = validate(model,test_ds,crit,validate_)

        print(f"epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}")



# def train_one_step(model,ds,crit,optim):
#     train_loss = 0
#     total = 0
#     correct = 0
    
#     for x,y in ds:
#         with tf.GradientTape() as tape:
#             o = model(x,training=True)
#             loss = crit(y,o)
#         gradients = tape.gradient(loss,model.trainable_variables)
#         optim.apply_gradients(zip(gradients,model.trainable_variables))
#         train_loss += loss
#         correct += np.sum(np.argmax(o,axis=1) == y)
#         total += len(y)
#     return train_loss, 100 * correct / total

# def validate(model,ds,crit):
#     test_loss = 0
#     total = 0
#     correct = 0
#     for x,y in ds:
#         o = model(x,training=False)
#         loss = crit(y,o)
#         test_loss += loss
#         total += len(y)
#         correct += np.sum(np.argmax(o,axis=1)==y)
#     return test_loss, 100 * correct / total

# def train(model,train_ds, test_ds, crit, optim, sched, epochs):
#     for epoch in range(epochs):
#         train_loss, train_acc = train_one_step(model,train_ds,crit, optim)
#         test_loss, test_acc = validate(model,test_ds,crit)

#         print(f"epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}")
