import csv
import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

MAX_DOCUMENT_LENGTH = 300
HIDDEN_SIZE = 250
MAX_LABEL = 4610
CHARS_FEATURE = 'chars'  # Name of the input character feature.

def char_rnn_model(features, labels, mode):
  """Character level recurrent neural network model to predict classes."""
  byte_vectors = tf.one_hot(features[CHARS_FEATURE], 256, 1., 0.)
  byte_list = tf.unstack(byte_vectors, axis=1)

  cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.contrib.rnn.static_rnn(cell, byte_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def batch_train(fn_list, pc_list, batchCounter):
    fn_array = np.array(fn_list, np.str)
    pc_array = np.array(pc_list, np.int)
    x_train = pandas.DataFrame(fn_array)[1]
    y_train = pandas.Series(pc_array)

    # Process vocabulary
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    # Train.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={CHARS_FEATURE: x_train},
        y=y_train,
        batch_size=len(x_train),
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=300)
    print('batch train: ' + str(batchCounter))

def predict(fn_list, pc_list):
    x_test = pandas.DataFrame(np.array(fn_list, np.str))[1]
    y_test = pandas.Series(np.array(pc_list, np.int))

    # Process vocabulary
    x_test = np.array(list(char_processor.fit_transform(x_test)))
    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={CHARS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy (sklearn): {0:f}'.format(score))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

def batch():
    with open('train_data.csv', newline='') as trainData:
        filename_list = []
        paper_code_list = []
        lineCounter = 0
        batchCounter = 1
        dataReader = csv.reader(trainData)
        for row in dataReader:
            if lineCounter < 500:
                paper_code_list.append(int(row[0]))
                fn = []
                fn.append('x')
                fn.append(row[1])
                filename_list.append(fn)
                lineCounter += 1
            else:
                batch_train(filename_list, paper_code_list, batchCounter)
                filename_list = []
                paper_code_list = []
                lineCounter = 0
                batchCounter += 1
    with open('train_data.csv', newline='') as predictData:
        p_list = []
        filename_list1 = []
        paper_code_list1 = []
        dataReader1 = csv.reader(predictData)
        for row in dataReader1:
            papercode = int(row[0])
            if papercode not in p_list:
                paper_code_list1.append(papercode)
                fn1 = []
                fn1.append('x')
                fn1.append(row[1])
                filename_list1.append(fn1)
        predict(filename_list1, paper_code_list1)

# Build model
classifier = tf.estimator.Estimator(model_fn=char_rnn_model, model_dir='model')
char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)

for count in range(0, 10):
    batch()