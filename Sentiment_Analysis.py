# ---------------------------------------------------importing libraries--------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

# Packages for data preparation
from sklearn.model_selection import train_test_split, cross_val_score
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers

# -----------------------------------------------------Constant Declaration-----------------------------------------------
NB_WORDS = 15000  
VAL_SIZE = 1000  
NB_START_EPOCHS = 10 
BATCH_SIZE = 512  

# -----------------------------------------------------Importing the dataset-----------------------------------------------
dataset = pd.read_csv('train.tsv', '\t')
dataset = dataset.reindex(np.random.permutation(dataset.index))
dataset = dataset.iloc[:20000, [2,3]]
dataset.head()

# --------------------------------------------------------Data cleaning----------------------------------------------------
def remove_stopwords(input_text):
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]
    print("input_text : {}".format(input_text))
    input_words = input_text.split()
    print("input_words : {}".format(input_words))
    clean_words = [input_word for input_word in input_words if (input_word not in stopwords_list 
                                                                or input_word in whitelist) and len(input_word) > 1]
    return " ".join(clean_words)

dataset.Phrase = dataset.Phrase.apply(remove_stopwords)

# ------------------------------------------------Train-Test split of dataset----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(dataset.Phrase, 
                                                    dataset.Sentiment,
                                                    test_size = 0.1, 
                                                    random_state = 0)
# Checking the shape of X and corresponding y
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]

# ------------------------------------------Encoding the features(tokenizing)-----------------------------------
tokenizer = Tokenizer(num_words = NB_WORDS,
                      filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = True, 
                      split = " ")
tokenizer.fit_on_texts(X_train)
print('Fitted tokenizer on {} documents'.format(tokenizer.document_count))
print('{} words in dictionary'.format(tokenizer.num_words))
print('Top 5 most common words are:', collections.Counter(tokenizer.word_counts).most_common(5))

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# One Hot Encoding
def one_hot_seq(seq, nb_features = NB_WORDS):
    ohs = np.zeros((len(seq), nb_features))
    print("ohs : {}".format(ohs))
    for i, s in enumerate(seq):
        print("i : {} s : {}".format(i, s))
        ohs[i, s] = 1
        print("ohs : {}".format(ohs))
        return ohs
    
X_train_oh = one_hot_seq(X_train_seq)
X_test_oh = one_hot_seq(X_test_seq)

y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)

X_train_rest, X_validate, y_train_rest, y_validate = train_test_split(X_train_oh, 
                                                                      y_train_oh,
                                                                      test_size = 0.1, 
                                                                      random_state = 0)

assert X_validate.shape[0] == y_validate.shape[0]
assert X_train_rest.shape[0] == y_train_rest.shape[0]

print('Shape of validation set:',X_validate.shape)

#-------------------------------------------------Building the model------------------------------
def deep_model(model):
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    
    history = model.fit(X_train_rest, 
                        y_train_rest,
                        epochs = NB_START_EPOCHS,
                        batch_size = BATCH_SIZE,
                        validation_data = (X_validate, y_validate),
                        verbose = 0)
    return history

def eval_metric(history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name] 

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel("No.of epochs")
    plt.legend()
    plt.show()
    
def test_model(model, epoch_stop):
    model.fit(X_train_oh
              , y_train_oh
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0)
    results = model.evaluate(X_test_oh, y_test_oh)
    return results

base_model = models.Sequential()
base_model.add(layers.Dense(64, activation='relu', input_shape = (NB_WORDS,)))
base_model.add(layers.Dense(64, activation='relu'))
base_model.add(layers.Dense(5, activation='softmax'))
base_model.summary()

base_history = deep_model(base_model)  
eval_metric(base_history, 'loss')
eval_metric(base_history, 'accuracy')
base_results = test_model(base_model, 10)

print('Test accuracy of baseline model: {0:.2f}%'.format(base_results[1]*100))

base_model.layers.pop()
base_model.layers.pop()
base_model.layers.pop()

reduced_model = models.Sequential()
reduced_model.add(layers.Dense(256, activation='relu', input_shape = (NB_WORDS,)))
reduced_model.add(layers.Dense(5, activation='softmax'))
reduced_model.summary()

reduced_history = deep_model(reduced_model)
eval_metric(reduced_history, 'accuracy')
reduced_results = test_model(reduced_model, 9)
print('Test accuracy of reduced model: {0:.2f}%'.format(reduced_results[1]*100))

reg_model = models.Sequential()
reg_model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(NB_WORDS,)))
reg_model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
reg_model.add(layers.Dense(5, activation='softmax'))
reg_model.summary()

reg_history = deep_model(reg_model)
eval_metric(reg_history, 'accuracy')
reg_results = test_model(reg_model, 10)
print('Test accuracy of regularized model: {0:.2f}%'.format(reg_results[1]*100))

drop_model = models.Sequential()
drop_model.add(layers.Dense(64, activation='relu', input_shape=(NB_WORDS,)))
drop_model.add(layers.Dropout(0.5))
drop_model.add(layers.Dense(64, activation='relu'))
drop_model.add(layers.Dropout(0.5))
drop_model.add(layers.Dense(64, activation='relu'))
drop_model.add(layers.Dropout(0.5))
drop_model.add(layers.Dense(64, activation='relu'))
drop_model.add(layers.Dropout(0.5))
drop_model.add(layers.Dense(5, activation='softmax'))
drop_model.summary()

drop_history = deep_model(drop_model)
eval_metric(drop_history, 'loss')
drop_results = test_model(drop_model, 12)
print('Test accuracy of dropout model: {0:.2f}%'.format(drop_results[1]*100))
