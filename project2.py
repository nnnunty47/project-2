import sys
import networkx as nx
import pandas as pd
import numpy as np
import json
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')
NUM = 500

def parse_config(argv):
    config = {'N': 0, 'ingredients': []}

    n = len(argv)
    i = 1
    while i < n:
        if argv[i] == '--N':
            config['N']  = int(argv[i + 1])
            i += 2

        elif argv[i] == '--ingredient':
            config['ingredients'].append(argv[i + 1])
            i += 2
        else:
            i += 1
    return config

def build_graph():
    #read edge info in srep00196-s2.csv
    graph_data = pd.read_csv('data/srep00196-s2.csv',skiprows=4,header=None)
    graph_data.columns = ['ingredient1','ingredient2','shared_compounds']
    #read recipes info in srep00196-s2.csv
    recipes = pd.read_csv('data/srep00196-s3.csv',skiprows=3,sep='\t')
    recipes.columns=['recipes']
    recipes['country'] = recipes['recipes'].apply(lambda x: x.split(',')[0])
    recipes['ingredients'] = recipes['recipes'].apply(lambda x: x.split(',')[1:])
    
    all_ingredients = set()
    for ingredients in recipes['ingredients']:
        for ingredient in ingredients:
            all_ingredients.add(ingredient)
    

    edge_num = graph_data.shape[0]
    #bulid graph
    G = nx.Graph()
    graph_dict = {} #graph with dict format
    for i in range(edge_num):
        #extract edge whose node is in srep00196-s3.csv
        
        if (graph_data['ingredient1'][i] in all_ingredients) and (graph_data['ingredient2'][i] in all_ingredients):
            G.add_edge(graph_data['ingredient1'][i],graph_data['ingredient2'][i])
            graph_dict[graph_data['ingredient1'][i],graph_data['ingredient2'][i]] = graph_data['shared_compounds'][i]
            graph_dict[graph_data['ingredient2'][i],graph_data['ingredient1'][i]] = graph_data['shared_compounds'][i]

    #get prevalance
    prevalance = {}
    ingredient_count = {}
    total = 0
    for ingredient in all_ingredients:
        ingredient_count[ingredient] = 0
    for ingredients in recipes['ingredients']:
        for ingredient in ingredients:
            total += 1
            ingredient_count[ingredient] += 1
    for ingredient in all_ingredients:
        prevalance[ingredient] = ingredient_count[ingredient] / total
    return G, graph_dict, prevalance, all_ingredients

    
def replace_flavour_ingredient(ingredient):
    if 'oil' in ingredient: #use olive_oil for all oil
        return 'olive_oil'
    return ingredient

def clean_ingredients(ingredients, nlp, all_ingredients):
    new_ingredients = []
    clean_ingredients = []
    for ingredient in ingredients:
        new_ingredient = replace_flavour_ingredient(ingredient)
        new_ingredient = '_'.join([token.lemma_ for token in nlp(re.sub('[^a-zA-Z]',' ',new_ingredient))]) #ignore '-' in ingreident
        if new_ingredient in all_ingredients:
            clean_ingredients.append(new_ingredient) 
    return clean_ingredients
    
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=10086)
    model = svm.LinearSVC()
    model = CalibratedClassifierCV(model, method='sigmoid')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model
    
def get_pd_ingredients(all_ingredients_list, ingredients):
    data = pd.DataFrame(columns = all_ingredients_list)
    new_row = [ingredient in ingredients for ingredient in all_ingredients_list]
    data = data.append(pd.Series(new_row, index=data.columns), ignore_index=True)
    return data
def get_yummly_data(all_ingredients):
    nlp = spacy.load('en_core_web_sm')
    with open('data/yummly.json', 'r', encoding = 'utf-8') as f:
        text = f.read()
        text = json.loads(text)
        yummly_data = pd.json_normalize(text)
    yummly_data = yummly_data.head(NUM)
    
    yummly_data['ingredients'] = yummly_data['ingredients'].map(lambda x: clean_ingredients(x, nlp, all_ingredients))
    all_ingredients_list = list(sorted(list(all_ingredients)))
    
    train_data = pd.DataFrame()
    
    for ingredient in all_ingredients_list:
        train_data[ingredient] = yummly_data['ingredients'].apply(lambda ingredients : ingredient in ingredients)
    model = train(train_data, yummly_data['cuisine'])
    return yummly_data, model, all_ingredients_list, train_data

def calc_cosine_similarity(idxs, dishes, data):
    cosines = []
    data_row = np.array(data.iloc[0, :])
    for index, row in dishes.iterrows():
        print(index)
        cosines.append((1 - cosine(np.array(dishes.iloc[index, :]), data_row), idxs.iloc[index]))
    return list(sorted(cosines, key = lambda x : [-x[0], x[1]]))
if __name__ == '__main__':
    config = parse_config(sys.argv)
    G, graph_dict, prevalance, all_ingredients = build_graph()
    yummly_data, model, all_ingredients_list, train_data = get_yummly_data(all_ingredients)
    data = get_pd_ingredients(all_ingredients_list, config['ingredients'])
    
    result = {}
    cuisine = model.predict(data)
    result['cuisine'] = cuisine
    scores = model.predict_proba(data)
    idx = 0
    
    while model.classes_[idx] != cuisine:
        idx += 1
    score = scores[0][idx]
    result['score'] = score
    cosines = calc_cosine_similarity(yummly_data['id'], train_data, data)
    result['closest'] = [{'id': id, 'score' : score} for score, id in cosines[:config['N']]]
    print(result)
    
    