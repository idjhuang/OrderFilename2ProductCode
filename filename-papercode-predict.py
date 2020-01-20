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

def predict(filename, papercode):
    global fail_num, low_prob_num
    x_test = pandas.DataFrame(np.array([filename], np.str))[1]
    y_test = pandas.Series(np.array([papercode], np.int))

    # Process vocabulary
    x_test = np.array(list(char_processor.fit_transform(x_test)))
    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={CHARS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    for p in predictions:
        predict_papercode = p['class']
        prob = p['prob'][predict_papercode]
        if predict_papercode != papercode:
            log.write(fn[1] + '\n')
            log.write('predict paper code: ' + str(predict_papercode) + ', prob: ' + str(prob) + ', paper code: ' + str(papercode) + ' fail\n')
#            print(fn[1])
#            print('predict paper code: ' + str(predict_papercode) + ', prob: ' + str(prob) + ', paper code: ' + str(papercode) + ' fail')
            fail_num = fail_num + 1
        if prob < 0.99:
            low_prob_num = low_prob_num + 1

# Build model
classifier = tf.estimator.Estimator(model_fn=char_rnn_model, model_dir='model_2')
char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
total_num = 0
low_prob_num = 0
fail_num = 0

with open('result.log', 'a') as log:
    with open('train_data.csv', newline='') as predictData:
#    paper_code_list = []
        lineCounter = 0
        dataReader = csv.reader(predictData)
        for row in dataReader:
            total_num = total_num + 1
            papercode = int(row[0])
#        if papercode not in paper_code_list:
#            paper_code_list.append(papercode)
            fn = []
            fn.append('x')
            fn.append(row[1])
            predict(fn, papercode)

    log.write('total num: ' + str(total_num) + ', fail num: ' + str(fail_num) + ', low num: ' + str(low_prob_num) + '\n')
    log.write('correct rate: ' + str(1 - fail_num / total_num) + ', low rate: ' + str(low_prob_num / total_num) + '\n')