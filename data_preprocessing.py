import collections
import random
import numpy as np

VOCABULARY_SIZE = 20000
### generate_batch ###
# This function generates the train data and label batch from the dataset.
#
### Parameters ###
# batch_size: the number of train_data,label pairs to produce per batch
# curr_batch: the current batch number.
# window_size: the size of the context
# data: the dataset
### Return values ###
# train_data: train data for current batch
# labels: labels for current batch
def generate_batch(batch_size, data_index, window_size, data ):
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
  return batch, labels

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window,data):
  global data_index
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
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


### build_dataset ###
# This function is responsible of generating the dataset and dictionaries.
# While constructing the dictionary take into account the unseen words by
# retaining the rare (less frequent) words of the dataset from the dictionary
# and assigning to them a special token in the dictionary: UNK. This
# will train the model to handle the unseen words.
### Parameters ###
# words: a list of words
# vocab_size:  the size of vocabulary
#
### Return values ###
# data: list of codes (integers from 0 to vocabulary_size-1).
#       This is the original text but words are replaced by their codes
# dictionary: map of words(strings) to their codes(integers)
# reverse_dictionary: maps codes(integers) to words(strings)
def build_dataset(words, vocab_size):






    global VOCABULARY_SIZE
    counts=dict()
    for item in words:

        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    count=list()
    count.extend(collections.Counter(words).most_common(VOCABULARY_SIZE - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary.get(word, 0)
        data.append(index)

    sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    D=counts.items()
    new_dict=sorted_counts[:VOCABULARY_SIZE]
    suma = 0
    for i in range(len(new_dict),len(sorted_counts),1):
    #print sorted_counts[i:]

        suma = suma+sorted_counts[i][1]
    unk=('UNK',suma);
    b=[unk]+new_dict
    data1 = list(range(0, len(b)))
    e=dict(zip(data1,b))#rev
    aB=list(zip(*b))
    reverse_dictionary=aB[0]
    dictionary=dict(zip(reverse_dictionary,data1))#dict






    data = [x+1 for x in data]
    return data,dictionary,reverse_dictionary

###
# Save embedding vectors in a suitable representation for the domain identification task
###
def save_vectors(vectors):
    np.savetxt('/home/adil/Desktop/NLP/NLP_HW01/final_embedding_dic.txt',vectors)


    ###FILL HERE###



# Reads through the analogy question file.
#    Returns:
#      questions: a [n, 4] numpy array containing the analogy question's
#                 word ids.
#      questions_skipped: questions skipped due to unknown words.
#
def read_analogies(file, dictionary):
    questions = []
    questions_skipped = 0
    with open(file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip())) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)
