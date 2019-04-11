import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import callbacks
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime
from tensorflow.python.lib.io import file_io
import argparse

def main(args):

    ##Setting up the path for saving logs
    logs_path = args.job_dir + '/logs/'

    ##Using the GPU
    with tf.device('/device:GPU:0'):
    
        currentDT = datetime.datetime.now()
        print (str(currentDT))
        df = pd.read_csv(pd.read_csv('gs://mhsiendata/data/cleanlyrics3.csv'))
        df.columns = ["index", 'text', 'genre']
        data = df[['text', 'genre']]
        
        
        tokenizer = Tokenizer(num_words = 10000, split=' ')
        tokenizer.fit_on_texts(data['text'].values)
        #print(tokenizer.word_index)  # To see the dicstionary
        X = tokenizer.texts_to_sequences(data['text'].values)
        X = pad_sequences(X)
        
        embed_dim = 128
        lstm_out = 300
        batch_size= 64

        ##Buidling the LSTM network
        
        model = Sequential()
        model.add(Embedding(10000, embed_dim,input_length = X.shape[1]))
        model.add(LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(7,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

        model.summary()
        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        Y = pd.get_dummies(data['genre']).values
        X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)

        X_test, X_valid, Y_test, Y_valid = train_test_split(X,Y, test_size = 0.20, random_state = 36)
        #Here we train the Network.

        model.fit(X_train, Y_train, batch_size =batch_size, epochs = 1, callbacks=[tensorboard],  verbose = 1,validation_data=(X_valid,Y_valid))

        score,acc = model.evaluate(X_valid, Y_valid, verbose = 1, batch_size = batch_size)
        print("Logloss score: %.2f" % (score))
        print("Validation set Accuracy: %.2f" % (acc))

        currentDT = datetime.datetime.now()
        print (str(currentDT))
   
        # Save model.t1 on to google storage
        model.save('model.test1')
        with file_io.FileIO('model.test1', mode='r') as input_f:
            with file_io.FileIO(job_dir + '/model.test1', mode='w+') as output_f:
                output_f.write(input_f.read())
        
##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    
    parser.add_argument(
        '--train-files',
        help='GCS file or local paths to training data',
        nargs='+',
        default='gs://michaelhsientest/cleanlyrics3.csv')
    ARGUMENTS, _ = parser.parse_known_args()

    main(ARGUMENTS)                