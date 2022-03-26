from multiprocessing.connection import Client
from flask import Flask
from threading import Thread
from itertools import cycle

import discord
import nltk, os
#nltk.download('punkt')
import pandas as pd


from nltk import word_tokenize,sent_tokenize

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np 
import tflearn
import tensorflow as tf
import random
import json
import pickle

app = Flask('')

@app.route('/')
def main():
  return "Your Bot Is Ready"

def run():
  app.run(host="0.0.0.0", port=8000)

def keep_alive():
  server = Thread(target=run)
  server.start()

status = cycle(['with Python','JetHub'])

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
               bag.append(1)
            else:
              bag.append(0)
    
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)



net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if [os.path.isfile(i) for i in ["model.tflearn.meta", "model.tflearn.index"]] == [True, True]:
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)


class MyClient(discord.Client):
    async def on_ready(self):
        print('Bot ready !')
        print(self.user.name)
        print(self.user.id)
        print('------')
    
    async def change_status():
        await Client.change_presence(activity=discord.Game(next(status)))

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        elif message.content == "Show me Gandhalf !":
            movies = pd.read_json('movies.json', encoding='UTF-8')
            #print(movies['Series_Title'][2])
            
            for i in range(len(movies)):
                if movies['Series_Title'][i] == "The Lord of the Rings: The Fellowship of the Ring":
                    await message.channel.send(movies[i].format(message))
        
        else:
           inp = message.content
           result = model.predict([bag_of_words(inp, words)])[0]
           result_index = np.argmax(result)
           tag = labels[result_index]
           
           if result[result_index] > 0.7:
               for tg in data["intents"]:
                   if tg['tag'] == tag:
                       responses = tg['responses']
                
               bot_response=random.choice(responses)
               await message.channel.send(bot_response.format(message))
           else:
               await message.channel.send("I didnt get that. Can you explain or try again.".format(message))


client = MyClient()
client.run("OTU0MDAxMjg1MDA5NTM5MTUz.YjMw7g.bi4jLfPWp1AqJYw9UBg6upp2bIk")