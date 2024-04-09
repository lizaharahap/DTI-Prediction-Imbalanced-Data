import random
import math
import copy
import time
import json
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import pandas as pd


class OVRS_ACO(object):
    def __init__(self,init_pheromone = 0.3,rho = 0.8,num_ant = 20,max_iter = 50 ,max_idem=20,kfold=5,random_state=None):        
        #parameter setting
        self.init_pheromone = init_pheromone #initial pheromone on all edges
        self.rho = rho #evaporation rate pheromone update
        self.num_ant = num_ant #number of ants
        self.max_iter = max_iter #max iteration ACO
        self.max_idem = max_idem
        self.kfold = 5
        
        #set random seed
        if random_state != None:
            self.random_state = random_state
            random.seed(random_state)
    
    def set_model(self,X_train, y_train, ovrs_target = 0, n_ovrs_target = 59, model = None, oversampler = None):
        #initiate model
        self.X_train = X_train.copy().reset_index(drop=True)
        self.y_train = y_train.copy().reset_index(drop=True)
        
        self.ovrs_target = ovrs_target
        self.n_ovrs_target = n_ovrs_target
        
        #set model
        if model != None:
            self.model = model
        else:
            self.model = RandomForestClassifier(random_state = self.random_state)
        
        #set oversampler
        if oversampler != None:
            self.oversampler = oversampler
        else:
            self.oversampler = SMOTE(sampling_strategy={self.ovrs_target:self.n_ovrs_target},
                                    random_state=self.random_state,
                                    k_neighbors=11,
                                    n_jobs=-1)
        
        #oversampling
        self.X_smote,self.y_smote = self.oversampling(self.X_train,self.y_train,target=ovrs_target,n_target=n_ovrs_target)
        
        #set pheromone matrix
        smote_ids = self.X_smote.index.values
        self.pheromone_matrix = pd.DataFrame()
        self.pheromone_matrix['smote_id'] = smote_ids
        self.pheromone_matrix['pheromone'] = self.init_pheromone
        self.pheromone_matrix = self.pheromone_matrix.set_index('smote_id')
    
    def oversampling(self,X_train,y_train,target,n_target):
        oversampled = self.oversampler
        X_smote, y_smote = oversampled.fit_resample(X_train.reset_index(drop=True), y_train.reset_index(drop=True))

        len_real_data = X_train.shape[0]
        return (X_smote[len_real_data:], y_smote[len_real_data:])
    
    def generate_solution(self,probability):
        smote_ids = self.X_smote.index.values
        
        num_minority = self.y_train.value_counts()[self.ovrs_target]
        num_majority = self.y_train.value_counts()[np.abs(self.ovrs_target-1)]
        num_chosen = int(num_majority-num_minority)
        
        chosen_ids = random.choices(smote_ids,probability,k=num_chosen)
        return chosen_ids
    
    def pheromone_update(self,pheromone_matrix):
        pheromone_matrix['pheromone'] = (1-self.rho*pheromone_matrix['pheromone'])+pheromone_matrix['delta']
        self.pheromone_matrix['pheromone'] = pheromone_matrix['pheromone'].values
    
    def init_delta_to_pheromone_matrix(self,pheromone_matrix):
        pheromone_matrix['delta'] = 0
        return pheromone_matrix
    
    def add_delta_to_pheromone_matrix(self,solution,pheromone_matrix,fitness):
        ids = copy.deepcopy(solution)
        for i in ids:
            pheromone_matrix.loc[i,"delta"] = pheromone_matrix.loc[i]['delta']+fitness 

        return pheromone_matrix
    
    def add_probability_to_pheromone_matrix(self,pheromone_matrix):
        sum_pheromone = pheromone_matrix['pheromone'].sum()
        pheromone_matrix['probability'] = pheromone_matrix['pheromone']/sum_pheromone
        return pheromone_matrix
    
    def construct_solution(self):
        best_y_train = self.y_train.copy()
        best_X_train = self.X_train.copy()
                
        kf = KFold(n_splits=self.kfold,random_state=self.random_state,shuffle=True)
        kf.get_n_splits(best_X_train)
        fold_results = []
        for train_index,test_index in kf.split(best_X_train):
            kf_X_train, kf_X_test = best_X_train.loc[train_index], best_X_train.loc[test_index]
            kf_y_train, kf_y_test = best_y_train.loc[train_index], best_y_train.loc[test_index]

            pipeline = make_pipeline(self.model)
            pipeline.fit(kf_X_train,kf_y_train)
            best_fitness = f1_score(kf_y_test,pipeline.predict(kf_X_test),pos_label=1)

            fold_results.append(best_fitness)

        best_fitness = np.mean(fold_results)
        
        fitness_history = [best_fitness]
        
        idem_counter = 0
        for i in range(self.max_iter): #iteration
            best_found_X_train = None
            best_found_y_train = None
            best_found_fitness = 0
            local_pheromone_matrix = self.pheromone_matrix.copy()
            local_pheromone_matrix = self.init_delta_to_pheromone_matrix(local_pheromone_matrix)
            local_pheromone_matrix = self.add_probability_to_pheromone_matrix(local_pheromone_matrix)
            for ant in range(self.num_ant): #step
                #initiate solution (population)
                chosen_ids = self.generate_solution(local_pheromone_matrix['probability'].values)
                chosen_X_smote = self.X_smote.copy().loc[chosen_ids]
                chosen_y_smote = self.y_smote.copy().loc[chosen_ids]
                
                #classification
                kf = KFold(n_splits=self.kfold,random_state=self.random_state,shuffle=True)
                kf.get_n_splits(self.X_train.copy())
                fold_results = []
                for train_index,test_index in kf.split(self.X_train.copy()):
                    kf_X_train, kf_X_test = self.X_train.copy().loc[train_index], self.X_train.copy().loc[test_index]
                    kf_y_train, kf_y_test = self.y_train.copy().loc[train_index], self.y_train.copy().loc[test_index]
                    
                    kf_new_X_train = pd.concat([kf_X_train.copy(),chosen_X_smote])
                    kf_new_y_train = pd.concat([kf_y_train.copy(),chosen_y_smote])

                    pipeline = make_pipeline(self.model)
                    pipeline.fit(kf_new_X_train,kf_new_y_train)
                    fitness = f1_score(kf_y_test,pipeline.predict(kf_X_test),pos_label=1)
                    
                    fold_results.append(fitness)
                
                fitness = np.mean(fold_results)
                
                if fitness > best_found_fitness:
                    best_found_fitness = fitness
                    
                    best_found_X_train = pd.concat([self.X_train.copy(),chosen_X_smote])
                    best_found_X_train = best_found_X_train.reset_index(drop=True)
                    best_found_y_train = pd.concat([self.y_train.copy(),chosen_y_smote])
                    best_found_y_train = best_found_y_train.reset_index(drop=True)
                    
                
                local_pheromone_matrix = self.add_delta_to_pheromone_matrix(chosen_ids,local_pheromone_matrix,fitness)
            
            #pheromone update
            self.pheromone_update(local_pheromone_matrix)
            
            fitness_history.append(best_found_fitness)
                        
            #checking best vs best found
            if best_found_fitness > best_fitness:
                best_fitness = best_found_fitness
                best_X_train = best_found_X_train.copy()
                best_y_train = best_found_y_train.copy()
                idem_counter = 0
            else:
                idem_counter += 1
                if idem_counter > self.max_idem:
                    return best_X_train,best_y_train,best_fitness,fitness_history
        
        return best_X_train,best_y_train,best_fitness,fitness_history