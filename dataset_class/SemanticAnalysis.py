from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from keras import Input,Model
from keras.layers import Dense
import pandas as pd

class SemanticAnalysis():
  def calculate_acsm(news_df, n):
    vectorizer = CountVectorizer() #used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
    X = vectorizer.fit_transform(news_df) #scale the training data and also learn the scaling parameters.


    n_cap = 1
    svd_model = TruncatedSVD(n_components=n_cap)
    X2 = svd_model.fit_transform(X)


    D = svd_model.components_ #The right singular vectors of the input data.
    T = X2
    S = np.diagflat(svd_model.singular_values_) #ndarray of shape (n_components,)

    X = np.dot(T, S)
    X = np.dot(X, D)
    c = np.dot(X, X.T)

    acsm = None
    sum = 0
    for i in c:
        for j in i:
            if j < 0:
                j = 0
            sum += j
        acsm = sum / n / n
    # print("-------------acsm----------------------")
    # print(acsm)

    return acsm

    def additem(list):
        if len(list) < 3:
            q = len(list) - 1
            str = '_1'
            str2 = '_2'
            item = ''.join([list[q], str])
            item_2 = ''.join([list[q], str2])
            list.append(item)
            list.append(item_2)

        return list

    def word2vec_train(index):
        list = []
        list.append(index[0])
        words = []

        additem(index)


        for word in index:
            if word not in words:
                words.append(word)


        word2int = {}
        for i,word in enumerate(words):
            word2int[word] = i
        print(word2int)



        WINDOW_SIZE=2
        data = []
        # for sentence in index:
        for idx,word in enumerate(index):
            for neighbor in index[max(idx - WINDOW_SIZE,0) : min(idx + WINDOW_SIZE,len(index))+ 1]:
                if neighbor != word:
                    data.append([word, neighbor])
    #                   print(data)
        df = pd.DataFrame(data,columns=['input','label'])



        ONE_HOT_DIM = len(words)
        #function to convert numbers to one hot vectors
        def to_one_hot_encoding(data_point_index):
            one_hot_encoding = np.zeros(ONE_HOT_DIM)
            one_hot_encoding[data_point_index] = 1
            return one_hot_encoding

        X = []
        Y = []

        for x, y in zip(df['input'],df['label']):
            X.append(to_one_hot_encoding(word2int[x]))
            Y.append(to_one_hot_encoding(word2int[y]))

        x_train = np.array(X)
        y_train = np.array(Y)


        #Defining the size of the embedding
        embed_size = 3
        xx = Input(shape=(x_train.shape[1],))
        yy = Dense(units=embed_size, activation='linear')(xx)
        yy =Dense(units=y_train.shape[1],activation='softmax')(yy)
        model = Model(inputs=xx, outputs=yy)
        model.compile(loss = 'categorical_crossentropy',optimizer = 'adam')
        #optimizing the network weights
        model.fit(x=x_train,y=y_train,
            batch_size=128,
            epochs=500
        )
        weights = model.get_weights()[0]


        for word in words:

            q = weights[df.get(word)]

            arr_new = np.sum(q, axis=1)
            arr_new = arr_new.tolist()

            list.append(arr_new)

            break
        return list