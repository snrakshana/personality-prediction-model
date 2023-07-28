from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

import pickle
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import json
import facebook
import requests

p=PorterStemmer()
stwrds=stopwords.words("english")
# print(stwrds)
def filtr(st):
    arr=[re.sub("[@,#]","",x) for x in st.split() if x not in stwrds]
    arr=[p.stem(x) for x in arr]
    return ' '.join(arr)

def home(request):
    print('im')
    if request.method == 'POST':
        print(request.POST['myField'])
        access_token = request.POST['myField']
        #access_token='EAAmsKmpvOqYBABVebhcZBRyJxOZC9p2TJyC6fZAgqlFAbMtZBxOACxxKyy9Vi0tT3BsDi0FKH7uvfp1BsQbVolkgZCllR4CvlEm5ZCsFCjPMqA5WBAzR5354gOFKbcwX144eke1KBptwVNZCNB7RLVsUJZAh2p94NWhjF3sQ6vL6OzpSWTrnFRfjZCRBZB2NXuZClUT5TGWH0DZBEQZDZD'
        graph = facebook.GraphAPI(access_token)
        #fields = ['message']
        posts = graph.request('/me/posts')
        count=1
        text=[]
        while "paging" in posts: 
            #print("length of the dictionary",len(posts))
            #print("length of the data part",len(posts['data']))
            for post in posts["data"]:
                if "message" in post:   #because some posts may not have a caption
                    print(post["message"]) 
                    text.append(filtr(post["message"]))
                    text.append("|||")
                count=count+1

            posts=requests.get(posts["paging"]["next"]).json()

        #print("end of posts")
        text=' '.join(text)
        print(text)

        tcv=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("static/input/vocab.pkl","rb")))
        m=load_model('static/input/mymodel.h5')
        res=np.argmax(m.predict(tcv.transform([text])))
        typ=['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
            'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

        my_typ = typ[res]
        print("Personality: "+ my_typ)
        print(" Introversion (I) \n Extroversion (E) \n Intuition (N) \n Sensing (S) \n Thinking (T) \n Feeling (F) \n Judging (J) \n Perceiving (P)")

    context = {}

    return render(request,'model/home.html', context)
