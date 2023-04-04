from riotwatcher import LolWatcher, ApiError
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np 
import tensorflow as tf
import pickle
from time import sleep
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

st.set_page_config(page_title="Streamlit-HWR", layout="wide")

NN_Model = tf.keras.models.load_model('Models/nn_model.h5')

with open("Models/DT_Model.pkl", 'rb') as file:  
    DT_Model = pickle.load(file)

with open('Models/LG_Model.pkl', 'rb') as file:  
    LG_Model = pickle.load(file)

with open('Models/SVM_Model.pkl', 'rb') as file:  
    SVM_Model = pickle.load(file)

with open('Models/XGB_Model.pkl', 'rb') as file:  
    XGB_Model = pickle.load(file)

sc = MinMaxScaler()

with open('scaler.pkl', 'rb') as file:  
    sc = pickle.load(file)

load_dotenv()
API_KEY = os.getenv('API_KEY')
watcher = LolWatcher(API_KEY)
region = 'NA1' 


def get_summoner_uid(name):
    summoner = watcher.summoner.by_name(region, name)
    return summoner["puuid"]


def get_data_from_match(player_uid):
    
    nn_prediction = []
    dt_prediction = []
    lg_prediction = []
    svm_prediction = []
    xgb_prediction = []
    actual = []

    match_list = watcher.match.matchlist_by_puuid('americas', player_uid, queue=420, type="ranked", count=10)
    
    for match in match_list:
        game_timeline = watcher.match.timeline_by_match('americas', match)
        game_details = watcher.match.by_id('americas', match)
        if game_details["info"]["participants"][0]["win"] == True:
            winner = 1
        else:
            winner = 0
                
       
        if len(game_timeline['info']['frames']) < 22:
            # print("less than 20")
            continue 
        # print("more than 20")
        frames =  game_timeline['info']['frames'][:22]
        
        # get gold diff 
        gold = {}
        for k in range(len(frames)):
            gold['{}'.format(k)] = {}
        for value in gold.values():
            for l in range(1, 11):
                value['{}'.format(l)] = 0

        for m in range(len(frames)):
            for n in range(1, 11):
                gold['{}'.format(m)]['{}'.format(n)] = game_timeline['info']['frames'][m]['participantFrames']['{}'.format(n)]['totalGold']

        blue_gold = []
        red_gold = []
        
        for key in gold.keys():
            team1 = 0
            team2 = 0
            for o in range(1, 6):
                team1 += gold[key]['{}'.format(o)]
                team2 += gold[key]['{}'.format(o + 5)]
            blue_gold.append(team1)
            red_gold.append(team2)

        # gold_diff = (np.array(team_1_gold) - np.array(team_2_gold)).tolist()
        
        
        blue_dragons = 0
        blue_heralds = 0
        red_dragons = 0
        red_heralds = 0
        blue_barons = 0
        red_barons = 0

        for frame in frames:
            for event in frame["events"]:
                if "monsterType" in event.keys():
                    if event["monsterType"] == "DRAGON":
                        if event["killerTeamId"] == 100:
                            blue_dragons += 1
                        else:
                            red_dragons += 1
                    if event["monsterType"] == "RIFTHERALD":
                        if event["killerTeamId"] == 100:
                            blue_heralds += 1
                        else:
                            red_heralds += 1
                    if event["monsterType"] == "BARON_NASHOR":
                        if event["killerTeamId"] == 100:
                            blue_barons += 1
                        else:
                            red_barons += 1
          
        # print(match, blue_gold[-1], red_gold[-1], blue_dragons, blue_heralds, blue_barons, red_dragons, red_heralds, red_barons)
        
        blue_k = 0 # blue side (bottom)
        red_k = 0 # red side (top)

        for frame in frames:
            for event in frame["events"]:
                if "victimId" in event.keys():
                    if event["type"] == "CHAMPION_KILL":
                        if event["killerId"] == 0:
                            continue
                        if 1 <= event["killerId"] <= 5:
                            blue_k += 1
                        else:
                            red_k += 1
                            
        blue_obj = 0
        red_obj = 0
        for frame in frames:
            for event in frame["events"]:
                if "buildingType" in event.keys():
                    if event["teamId"] == 100:
                        red_obj += 1
                    else:
                        blue_obj += 1
        
        tensor = [blue_dragons, blue_heralds, blue_barons, blue_obj, blue_k, blue_gold[-1], red_dragons, red_heralds, red_barons, red_obj, red_k, red_gold[-1]]
        
        input_data = sc.transform([tensor])
        nn_prediction.append(round(NN_Model.predict(input_data)[0][0]))
        dt_prediction.append(DT_Model.predict(input_data)[0])
        xgb_prediction.append(XGB_Model.predict(input_data)[0])
        svm_prediction.append(SVM_Model.predict(input_data)[0])
        lg_prediction.append(LG_Model.predict(input_data)[0])
        actual.append(winner)

    data = {
        'Neural Network': nn_prediction,
        'Decision Tree': dt_prediction,
        'X Gradient Boost': xgb_prediction,
        'Support Vector Machine': svm_prediction,
        'Logistic Regression': lg_prediction,
        'Actual Outcome': actual
    }

    return(data)

try:
    ign = st.text_input("Enter a summoner name")
    if not ign:
        st.warning("Please enter a name")
        st.stop()
except ApiError as err:
    if err.response.status_code == 404:
        st.warning("Please enter a valid summoner name")
        st.stop()
    else:
        raise


me = get_summoner_uid(ign)

df = pd.DataFrame.from_dict(get_data_from_match(me)) 

st.write(df)


