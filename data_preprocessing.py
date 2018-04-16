import collections
import random
import numpy as np

VOCABULARY_SIZE = 20000
data_index = 0

## This is the same tensorflow implementation which i studied and rebuilded
### I followed the same method but i re wrote i again after understanding
def generate_batch(batch_size, num_skips, skip_window,data):

  batch = np.ndarray(shape=(batch_size)) # creating an array to store the batch and labels
  labels = np.ndarray(shape=(batch_size, 1))
  span = 2 * skip_window + 1  # this is the no.of words that have to be found so right+left+present word
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

# I builded this fuction by refering to tf example
def build_dataset(words, vocab_size):

global VOCABULARY_SIZE
    counts=dict() # creating an empty dictionary to store the number of counts of each words

    # finding the cound of each words in the words
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
    # arraging the counts in increasing to decreasing order
    sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    D=counts.items()
    # count only till vocab size and rest of the will be treated as UNK
    new_dict=sorted_counts[:VOCABULARY_SIZE]
    suma = 0# a variable to store the count of UNK
    for i in range(len(new_dict),len(sorted_counts),1):
    #print sorted_counts[i:]
        # for all words other than vocab size, count of UNK will be increased by 1
        suma = suma+sorted_counts[i][1]
    # an ordered pair of UNK and the no.of UNKs
    unk=('UNK',suma);
    #Appending the UNK to the exsiting dictionary in a sorted place
    b=[unk]+new_dict

    data1 = list(range(0, len(b)))
    e=dict(zip(data1,b))#rev
    # making a dictionary with an index
    aB=list(zip(*b))
    # the reverse dictionary is the coloumn without the index
    reverse_dictionary=aB[0]

    # data is the index no.of each words before making the sort
    #so the dictionary will be in that format
    dictionary=dict(zip(reverse_dictionary,data1))#dict

    data = [x+1 for x in data]
    return data,dictionary,reverse_dictionary

###
# Save embedding vectors in a suitable representation for the domain identification task
###
def save_vectors(vectors):
    #just saving the entire embedding to a txt file, later i will compare with the metadata file and get each embeddings
    np.savetxt('/home/adil/Desktop/NLP/NLP_HW01/embd.txt',vectors)



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
