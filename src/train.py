# coding utf-8
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import utils
import functions
import neuralnet


# load dataset
dataset = utils.mnist_dataset("data")
X_train, X_test, y_train, y_test = dataset.load()

print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)

# create validation data
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=True
)

############## train model ##############
# parameters
epochs = 100
learning_rate = 0.1
batch_size = 64
patience_n_epoch = 5

h1_nodes = 256
h2_nodes = 128

# define the network structure
model = neuralnet.MLP_MNIST(
    input_nodes=784,
    h1_nodes=h1_nodes,
    h2_nodes=h2_nodes,
    output_nodes=10,
    learning_rate=learning_rate,
)

# calculate batch number per epoch
if len(y_train) % batch_size == 0:
    batch_per_epoch = len(y_train) // batch_size
else:
    batch_per_epoch = len(y_train) // batch_size + 1

# parameters for early stopping
best_valid_loss = np.inf
cnt = 0

for epoch in range(epochs):
    batch_generator = utils.batch_generator(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    train_loss_epoch = np.array([])

    for ii in range(batch_per_epoch):
        X_batch, y_batch = next(batch_generator)
        train_loss_batch = model.train(X_batch, y_batch)
        train_loss_epoch = np.append(train_loss_epoch, train_loss_batch)
        # display training loss per 100 batches
        if ii % 100 == 0:
            print(
                "Epoch: {}/{}, ".format(epoch + 1, epochs),
                "n_batch: {0}, ".format(ii),
                "Training loss in last batch: {:.4f}".format(train_loss_batch),
            )

    # calculate the average training loss
    avg_train_loss = np.average(train_loss_epoch)
    # calculate the validation loss
    valid_loss = model.evaluate(X_valid, y_valid)
    # display training results
    print(
        "--- Epoch: {}/{} finished, ".format(epoch + 1, epochs),
        "avg_train_loss in epoch, : {:.4f}".format(avg_train_loss),
        "avg_valid_loss of one data: {:.4f} ---".format(valid_loss),
    )

    # define the rules for early stopping
    if valid_loss < best_valid_loss:
        print(
            "--- Validation loss improved from {0:.4f} to {1:.4f} ---".format(
                best_valid_loss, valid_loss
            )
        )
        best_valid_loss = valid_loss
        cnt = 0
        # save model
        model.save_model("saved_data")
    else:
        print("Validation loss didn't improved")
        cnt += 1
        if cnt >= patience_n_epoch:
            print("--- Stop training ---")
            break

############## evaluate model using test data ##############

model_test = neuralnet.MLP_MNIST(
    input_nodes=784,
    h1_nodes=h1_nodes,
    h2_nodes=h2_nodes,
    output_nodes=10,
    learning_rate=learning_rate,
)
model_test.load_model("saved_data")

# predict probability
y_pred_test = model_test.predict(X_test)
print(y_pred_test)
test_loss = model_test.cross_entropy_loss(y_test, y_pred_test)
# labels
y_test_label = np.argmax(y_test, axis=1)
y_pred_label = np.argmax(y_pred_test, axis=1)
# evaluate model
print("Test loss: ", test_loss)
print(
    "Classification report:\n %s" % (classification_report(y_test_label, y_pred_label))
)
print("Confusion matrix:\n%s" % confusion_matrix(y_test_label, y_pred_label))
print("Accuracy: {}".format(accuracy_score(y_test_label, y_pred_label)))
