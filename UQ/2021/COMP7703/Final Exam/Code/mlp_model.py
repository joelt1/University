import os
from SupportCode import helpers
from math import sqrt
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

def MLPModel(data, dataset, topology={}, optimiser={}, act=tf.nn.relu, max_steps=100, path="", display_results=False, seed=None):
  # Set seed for reproducable results
  if seed:
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # CAUTION - calling function within loop will reset numpy seed everytime and affect RNG for Q2, Q3, Q4
    # np.random.seed(seed)
    tf.random.set_random_seed(seed)

  # Set up data
  x_train, y_train, x_valid, y_valid, x_test, y_test = data

  # Create inputs
  # tf.reset_default_graph()
  sess = tf.InteractiveSession()
  optimise = helpers.optimiserParams(optimiser)
  if optimise==None:
    print("Invalid Optimiser")
    return
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, x_train.shape[1]], name='x-input')
    y_labels = tf.placeholder(tf.float32, [None, y_train.shape[1]], name='y-input')

  # Generate hidden layers
  layers={}
  # Default of 1 hidden layer with 500 neurons if no argument passed to topology parameter
  hiddenDims = topology.setdefault("hiddenDims",[500])
  for i in range(len(hiddenDims)):
    if i==0:
      layers[str(i)] = helpers.FCLayer(x, x_train.shape[1], hiddenDims[i],"hidden_layer_"+str(i),act=act)
    else:
      layers[str(i)] = helpers.FCLayer(layers[str(i-1)],hiddenDims[i-1],hiddenDims[i],"hidden_layer_"+str(i),act=act)
  y = helpers.FCLayer(layers[str(i)], hiddenDims[i], y_train.shape[1], 'output_layer', act=tf.identity)


  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)
  with tf.name_scope('train'):
    train_step = optimise.minimize(cross_entropy)
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_labels, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()

  # Save results (disabled)
  """
  if not path:
    trainPath,testPath = helpers.getSaveDir(dataset, optimiser["optMethod"])
  else:
    trainPath,testPath = helpers.getSaveDir(dataset, path)
  train_writer = tf.summary.FileWriter(trainPath, sess.graph)
  test_writer = tf.summary.FileWriter(testPath)
  """
  
  tf.global_variables_initializer().run()
  
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      # Randomly choose 100 data samples to train on (x_train and y_train)
      idx = np.random.randint(0, len(x_train), 100)
      xs = [x_train[i] for i in idx]
      ys = [y_train[i] for i in idx]
      
    else:
      xs, ys = x_valid, y_valid
    return {x: xs, y_labels: ys}

  for i in range(max_steps):
    try:
      if i % 50 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        # Saving test results (disabled)
        # test_writer.add_summary(summary, i)
        print(('Accuracy at step %s: %s' % (i, acc)))
      else:  # Record train set summaries, and train
        if i % 25 == 24:  # Record execution stats
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary, _ = sess.run([merged, train_step],
                                feed_dict=feed_dict(True),
                                options=run_options,
                                run_metadata=run_metadata)
          # Save train results (disabled)
          # train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          # train_writer.add_summary(summary, i)
          # print(('Adding run metadata for', i))
        else:  # Record a summary
          summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
          # Save train results (disabled)
          # train_writer.add_summary(summary, i)
    except:
      print("FAILED DURING TRAINING")
      break

  # Close saved results file (disabled)
  # train_writer.close()
  # test_writer.close()

  # Validation set accuracy
  valid_accuracy = sess.run(accuracy, feed_dict={x: x_valid, y_labels: y_valid})
  print(f"Accuracy on valid set: {valid_accuracy}")
  print("\n")

  # Also get model test set accuracy and predictions on test set
  test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_labels: y_test})
  predictions = sess.run(tf.argmax(y, 1), feed_dict={x: x_test})
  sess.close()

  # Display results (won't work if results aren't saved)
  if display_results:
    # portTrain=8001, portTest=8002
    helpers.openTensorBoard(trainPath,testPath)

  return valid_accuracy, test_accuracy, predictions
