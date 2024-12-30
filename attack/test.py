import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import json
import random
from keras import models

nltk.download('punkt_tab')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
model = models.load_model('D:\\Dual-edge-defence-and-offensive-cyber-intelligence\\attack\\chatbot_model.h5')


words = pickle.load(open('D:\\Dual-edge-defence-and-offensive-cyber-intelligence\\attack\\words.pkl', 'rb'))
classes = pickle.load(open('D:\\Dual-edge-defence-and-offensive-cyber-intelligence\\attack\\classes.pkl', 'rb'))

with open('D:\\Dual-edge-defence-and-offensive-cyber-intelligence\\attack\\intents.json') as json_data:
    intents = json.load(json_data)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words   

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Chatbot is ready to talk! (type 'quit' to stop)")

while True:
    message = input("")
    if message.lower() == "quit":
        break

    intents = predict_class(message, model)
    response = get_response(intents, intents)
    print(response)
