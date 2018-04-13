import os
import collections
import random
import tensorflow as tf
import numpy as np
import tqdm
import re
import math
from tensorflow.contrib.tensorboard.plugins import projector
from data_preprocessing import build_dataset, save_vectors, read_analogies
from evaluation import evaluation
from gensim.parsing.porter import PorterStemmer

# run on CPU
# comment this part if you want to run it on GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

### PARAMETERS ###

BATCH_SIZE = 300 #3.4,300Number of samples per batch.RICoRDA CHE HAI STABILITO CHE BATCH%2*WIND=0
EMBEDDING_SIZE = 128# Dimension of the embedding vector.
WINDOW_SIZE = 5  # How many words to consider left and right.
NEG_SAMPLES = 180  # Number of negative examples to sample.
VOCABULARY_SIZE = 20000#60000 #The most N word to consider in the dictionary

TRAIN_DIR = "/home/adil/NLP/hw_1/materials/code/dataset/DATA/TRAIN"
VALID_DIR = "/home/adil/NLP/hw_1/materials/code/dataset/DATA/DEV"
TMP_DIR = "/tmp/"
ANALOGIES_FILE = "/home/adil/NLP/hw_1/materials/code/dataset/eval/questions-words.txt"


data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window,data):
  global data_index
  #print("in"+str(data_index))
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  #print("l" + str(data_index))
  data_index = (data_index + len(data) - span) % len(data)
  #print("o"+str(data_index))
  return batch, labels
""""def generate_batch(batch_size, data_index, window_size, data ):
  curr_batch = 2 * window_size # modo migliore per visualizzrli quando =
  assert batch_size % curr_batch == 0
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * window_size + 1  # [ window_size target window_size ]
  buffer = collections.deque(maxlen=span)
  data_index = (data_index + len(data) - span) % len(data)
  for _ in range(span):
    #print(data_index)
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // curr_batch):
    target = window_size  # target label at the center of the buffer
    targets_to_avoid = [window_size]
    for j in range(curr_batch):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * curr_batch + j] = buffer[window_size]
      labels[i * curr_batch + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels"""

### READ THE TEXT FILES ###

def get_latin(line):
    return ' '.join(''.join([i if ord(i) >=65 and ord(i) <=90 or  ord(i) >= 97 and ord(i) <= 122 else ' ' for i in line]).split())

p = PorterStemmer()

stoplist = set([w.rstrip('\r\n') for w in open("stopwords.txt")])
# Read the data into a list of strings.
# the domain_words parameters limits the number of words to be loaded per domain

def read_data(directory, domain_words=-1):
    i=1
    data = []
    for domain in os.listdir(directory):

        print(domain)
        i=i+1
        limit = domain_words
        for f in os.listdir(os.path.join(directory, domain)):
            #print(f)
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f)) as file:
                    for line in file.readlines():
                        #line=line.strip().lower()
                        #line=p.stem_sentence(line)
                        #line=line.lower()
                        line = line.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
                        line=re.sub("(^|\W)\d+($|\W)", " ", line)
                        line = get_latin(line)
                        split = line.lower().strip().split()
                        split = [word for word in split if (word not in stoplist) and (len(word)>1)]
                        if limit > 0 and limit - len(split) < 0:
                            split = split[:limit]
                        else:
                            limit -= len(split)
                        if limit >= 0 or limit == -1:
                            data += split
                        #print(data)
    return data

# load the training set
raw_data = read_data(TRAIN_DIR, domain_words=20000)#40000#ricorda che domainw limita il massimo numero di parole che possono essere lette
print('Data size', len(raw_data))
#print(raw_data)
# the portion of the training set used for data evaluation
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
#valid_examples_m =["german", "most","general", "food", "cat", "eat", "teach"]
#print(valid_examples)


### CREATE THE DATASET AND WORD-INT MAPPING ###

data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)
print('Dictionary size',len(dictionary))
del raw_data  # Hint to reduce memory.
# read the question file for the Analogical Reasoning evaluation
questions = read_analogies(ANALOGIES_FILE, dictionary)
#print(questions)

### MODEL DEFINITION ###

graph = tf.Graph()
eval = None

with graph.as_default():
    # Define input data tensors.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    ### FILL HERE ###
    #definisco le due matrici W1 e W2
    embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0)) #placeholder variable
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    nce_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE],stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
    nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))
    ### FILL HERE ###

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=NEG_SAMPLES,
                           num_classes=VOCABULARY_SIZE))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer =  tf.train.GradientDescentOptimizer(0.3).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

    # evaluation graph
    eval = evaluation(normalized_embeddings, dictionary, questions)

### TRAINING ###

# Step 5: Begin training.
num_steps = 30000001

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR, session.graph)
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    bar = tqdm.tqdm(range(num_steps))
    for step in bar:
        batch_inputs, batch_labels = generate_batch(BATCH_SIZE,2*WINDOW_SIZE, WINDOW_SIZE, data)
        #print({train_inputs: batch_inputs, train_labels: batch_labels})
        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict={train_inputs: batch_inputs, train_labels: batch_labels},
            run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                #print(valid_examples[i],reverse_dictionary[valid_examples[i]])
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                #print(nearest)
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)
        if step % 10000 == 0:
            if step>0:
                eval.eval(session)
                print("avg loss: "+str(average_loss / step))
    final_embeddings = normalized_embeddings.eval()
    #print (final_embeddings[0])

    ### SAVE VECTORS ###
    #print(final_embeddings)



    # Write corresponding labels for the embeddings.
    with open('/home/adil/Desktop/test/dict.tsv', 'w') as f:
        for i in range(VOCABULARY_SIZE):
            f.write(reverse_dictionary[i]+'\n')




    save_vectors(final_embeddings)
    # Save the model for checkpoints
    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(TMP_DIR, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()
