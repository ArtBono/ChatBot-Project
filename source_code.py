#region Imports
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

import re
from dotenv import load_dotenv
import os
#endregion

#region DeepL Greetings
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
#endregion

#region researches in movies db
movies = pd.read_json('movies.json', encoding='UTF-8')

columns = ["Poster_Link","Series_Title","Released_Year","Certificate","Runtime",
           "Genre","IMDB_Rating","Overview","Meta_score","Director",
           "Star1","Star2","Star3","Star4","No_of_Votes","Gross"]

# find if a title is contained in movies db
def titre(msg):
    for mov in range(len(movies)):
        if re.search(movies['Series_Title'][mov].lower(),msg.lower()) != None:
            return movies['Series_Title'][mov]
    else:
        return 'not found'

# find a random title movie in a given genre
def genre(msg):
    listmovies = []
    for mov in range(len(movies)):
        listgenre = movies['Genre'][mov].split(', ')
        for g in listgenre:
            if re.search(g.lower(),msg.lower()) != None:
                listmovies.append([movies['Series_Title'][mov], movies['Poster_Link'][mov]])
    if listmovies != []:
        return listmovies[random.randint(0,len(listmovies))]
    else:
        return 'not found'

# find all the movies associated to a director
def director(msg):
    listmovies = []
    for mov in range(len(movies)):
        if re.search(movies['Director'][mov].lower(),msg.lower()) != None:
            listmovies.append("Title : " + movies['Series_Title'][mov] + "\t Genre : " + movies['Genre'][mov] +"\n")
    if listmovies != []:
        return listmovies
    else:
        return 'not found'

# find all the movies associated to an actor
def actor(msg):
    listmovies = []
    for mov in range(len(movies)):
        if re.search(movies['Star1'][mov].lower(),msg.lower()) != None or re.search(movies['Star2'][mov].lower(),msg.lower()) != None or re.search(movies['Star3'][mov].lower(),msg.lower()) != None or re.search(movies['Star4'][mov].lower(),msg.lower()) != None:
            listmovies.append("Title : " + movies['Series_Title'][mov] + "\t Genre : " + movies['Genre'][mov] +"\n")
    if listmovies != []:
        return listmovies
    else:
        return 'not found'

# find all the movies associated to a year
def year(msg):
    listmovies = []
    for mov in range(len(movies)):
        if re.search(movies['Released_Year'][mov],msg) != None:
            listmovies.append("Title : " + movies['Series_Title'][mov] + "\t Genre : " + movies['Genre'][mov] +"\n")
    if listmovies != []:
        return listmovies
    else:
        return 'not found'

#endregion

#region Client
class MyClient(discord.Client):
    async def on_ready(self):
        print('Bot ready !')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        # display intels about the queried movie
        elif titre(message.content) != "not found":
            for i in range(len(movies)):
                if movies['Series_Title'][i] == titre(message.content):
                    answer = [(elem + ' : ' + str(movies[elem][i])) for elem in columns]
                    for ans in answer :
                        await message.channel.send((ans + "\n").format(message))
            await message.channel.send('If you want to discover a movie in a particular category, just tell me which one.'.format(message))
        
        # display a title about an queried genre
        elif genre(message.content) != "not found":
            for el in genre(message.content):
                await message.channel.send(str(el).format(message))
            await message.channel.send("If you want to know about a particular director's filmography, just tell me her/his name.".format(message))

        # display the list of a director's movies
        elif director(message.content) != 'not found':
            for el in director(message.content):
                await message.channel.send(el.format(message))
            await message.channel.send("If you want to know about a particular actor's filmography, just tell me her/his name.".format(message))
        
        # display the list of an actor's movies
        elif actor(message.content) != 'not found':
            for el in actor(message.content):
                await message.channel.send(el.format(message))
            await message.channel.send('If you want to know about the movies realesed in a particular year, just tell me which one.'.format(message))
        
        # display the list of a year's movies
        elif year(message.content) != 'not found':
            for el in year(message.content):
                await message.channel.send(el.format(message))
            await message.channel.send('If you want to know about this movie or another in particular, just tell me the title.'.format(message))
        
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
               await message.channel.send("I didnt get that. Give me another movie title or try again.".format(message))
#endregion

client = MyClient()
load_dotenv(dotenv_path="config")
client.run(os.getenv("TOKEN"))