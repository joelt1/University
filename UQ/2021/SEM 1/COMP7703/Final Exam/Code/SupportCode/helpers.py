import os
import webbrowser
import subprocess
import signal
import platform
import re
from math import sqrt
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def convParams(topology):
  """
  Sets the convNet defaults
  """
  topology.setdefault('convPoolLayers',2)
  topology.setdefault('filterSize',5)
  topology.setdefault('convStride',1)
  topology.setdefault('numFilters',32)
  topology.setdefault('poolK',2)
  topology.setdefault('poolStride',2)
  topology.setdefault('FCLayerSize',1024)
  return topology

def optimiserParams(optDic):
  """
  Sets the default optimisation parameters
  """
  validOptimisers =["GradientDescent","Adam","RMSProp",
  "Momentum","Adagrad"]
  optimisation = {}
  opt=optDic.setdefault('optMethod',"GradientDescent")
  if opt not in validOptimisers:
    return None
  optimisation['learning_rate']=optDic.setdefault('learning_rate',0.001)
  if opt == "GradientDescent":
    optimiser=tf.train.GradientDescentOptimizer(**optimisation)
  elif opt == "Momentum":
    optimisation["momentum"]=optDic.setdefault('momentum',0.9)
    optimiser = tf.train.MomentumOptimizer(**optimisation)
  elif opt=="Adagrad":
    optimisation["initial_accumulator_value"]=optDic.setdefault('initial_accumulator_value',0.1)
    optimiser = tf.train.AdagradOptimizer(**optimisation)
  elif opt=="RMSProp":
    optimisation["momentum"]=optDic.setdefault('momentum',0.0)
    optimisation["decay"]=optDic.setdefault('decay',0.9)
    optimisation["centered"]=optDic.setdefault('centered',False)
    optimiser = tf.train.RMSPropOptimizer(**optimisation)
  elif opt=="Adam":
    optimisation["beta1"]=optDic.setdefault('beta1',0.9)
    optimisation["beta2"]=optDic.setdefault('beta2',0.999)
    optimiser = tf.train.AdamOptimizer(**optimisation)
  return optimiser

def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def dropoutLayer(inputs):
  with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(inputs, keep_prob)
    return dropped

def FCLayer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
      # Disabled since conv model disabled
      """
      # Visualize conv1 features
      with tf.variable_scope('heatmap'):
        # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
        x_min = tf.reduce_min(weights)
        x_max = tf.reduce_max(weights)
        weights_0_to_1 = (weights - x_min) / (x_max - x_min)
        weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
        # to tf.image_summary format [batch_size, height, width, channels]
        weights_transposed = tf.reshape(weights_0_to_255_uint8, [-1, int(weights_0_to_255_uint8.shape[0]), 
          int(weights_0_to_255_uint8.shape[1]), 1])
        # this will display random 3 filters from the 64 in conv1
        tf.summary.image('heatmap', weights_transposed,10)
      """
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def killProcessesOnPorts(portTrain,portTest):
    ports=[str(portTrain),str(portTest)]
    if "Windows" in platform.system():
        popen = subprocess.Popen(['netstat', '-a','-n','-o'],
                           shell=False,
                           stdout=subprocess.PIPE)
    else:
        popen = subprocess.Popen(['netstat', '-lpn'],
                         shell=False,
                         stdout=subprocess.PIPE)
    (data, err) = popen.communicate()
    data = data.decode("utf-8")
    
    if "Windows" in platform.system():
        for line in data.split('\n'):
            line = line.strip()
            for port in ports:
                if '127.0.0.1:' + port in line and "0.0.0.0:" in line:
                    pid = line.split()[-1]
                    subprocess.Popen(['Taskkill', '/PID', pid, '/F'])
    else:
        pattern = "^tcp.*((?:{0})).* (?P<pid>[0-9]*)/.*$"
        pattern = pattern.format(')|(?:'.join(ports))
        prog = re.compile(pattern)
        for line in data.split('\n'):
            match = re.match(prog, line)
            if match:
                pid = match.group('pid')
                subprocess.Popen(['kill', '-9', pid])

def openTensorBoard(trainPath,testPath,portTrain=8001,portTest=8002):
  urlTrain = 'http://localhost:'+str(portTrain)
  urlTest = 'http://localhost:'+str(portTest)
  
  killProcessesOnPorts(portTrain,portTest)
  proc = subprocess.Popen(['tensorboard', '--logdir=' + trainPath,'--host=localhost', '--port=' + str(portTrain)])
  proc2 = subprocess.Popen(['tensorboard', '--logdir=' + testPath,'--host=localhost', '--port=' + str(portTest)])
  # time.sleep(4)
  webbrowser.open(urlTrain)
  webbrowser.open(urlTest)

def getSaveDir(dataset, path):
  directory = os.path.dirname(os.path.abspath(__file__))
  if "Windows" in platform.system():
    directory="/".join(directory.split("\\"))
  directory=directory.rsplit("/",1)[0] + "/Results/" + dataset + "/" + path + "/"
  if not os.path.exists(directory):
    os.makedirs(directory)
  dirList = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
  dirName = "train"
  maxNum = -1
  for d in dirList:
    if dirName in d:
      num = list(map(int, re.findall('\d+', d)))
      if len(num)==1:
          if num[0]>maxNum:
              maxNum=num[0]
  trainDir = directory + dirName + str(maxNum+1) +'/'
  testDir = directory + "test" + str(maxNum+1) + '/'
  return trainDir, testDir

def openTensorBoardAtIndex(dataset, path, ind, portTrain=8001, portTest=8002):
  directory = os.path.dirname(os.path.abspath(__file__))
  if "Windows" in platform.system():
    directory="/".join(directory.split("\\"))
  directory=directory.rsplit("/",1)[0] + "/Results/" + dataset + "/" + path + "/"
  if not os.path.exists(directory):
    print("You need to run at least a model on the given dataset")
    return
  dirList = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
  dirName = "train"
  foundDir = False
  for d in dirList:
    if dirName in d:
      num = list(map(int, re.findall('\d+', d)))
      if len(num)==1:
          if num[0]==ind:
              foundDir=True
  if not foundDir:
    print("That index does not exist")
    return
  else:
    trainDir = directory + dirName + str(ind) +'/'
    testDir = directory + "test" + str(ind) + '/'
    openTensorBoard(trainDir,testDir,portTrain,portTest)
