#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:28:17 2020

@author: ahmedsaad3
"""
import pandas as pd
import math
import collections
from pandas.api.types import is_numeric_dtype
from sklearn import tree
from sklearn.metrics import classification_report

class C45_Tree:
  __dictionary_tree={}
  __threeshold_dictionary={}
  __categorized_dictionary_by_target={}
  def __init__(self,input_data_frame, label_of_target_class):
    self.__training_data = input_data_frame
    self.__label_of_target_class = label_of_target_class
    
  def fit(self):
      self.__dictionary_tree={}
      self.__threeshold_dictionary=self.__prepare_data(self.__training_data)
      X_data=self.__training_data.loc[:,self.__training_data.columns!= self.__label_of_target_class]
      y_data=self.__training_data[self.__label_of_target_class]
      self.__categorized_dictionary_by_target=self.__build_categorized_dictionaries(X_data,y_data)[1]
      self.__build_tree(self.__training_data,self.__dictionary_tree)
      
  def test(self,testing_data):
    predicted_data=self.predict(testing_data)
    count_of_correct_values=0
    for i in range(len (predicted_data)):
        if testing_data.iloc[i, testing_data.columns.get_loc(self.__label_of_target_class)]==predicted_data[i]:
            count_of_correct_values+=1
    return count_of_correct_values,predicted_data

  def predict(self,data_frame):
        self.__update_testing_data(data_frame,self.__threeshold_dictionary)
        predicted_data=[]
    
        for row in range(len(data_frame)):
            not_found=False
            node=list(self.__dictionary_tree.keys())[0]
            value=data_frame.iloc[row, data_frame.columns.get_loc(node)]
            branch=self.__dictionary_tree[node][value]
            while(True):
                node=list(branch.keys())[0]
                if node=='leaf':
                    break
                value=data_frame.iloc[row, data_frame.columns.get_loc(node)]
                if(value not in branch[node] ):
                    not_found=True
                    break;
                branch=branch[node][value]
            if not_found:
                predicted_data.append(self.__get_max_class(self.__training_data))
            else:
                predicted_data.append(branch[node])
        return predicted_data
  
  def __build_tree(self,data_frame,dictionary_tree,attr_label=None,attr_value=None):
    if data_frame.empty:
        return
        
    if(len(data_frame.columns)==1):
        dictionary_tree['leaf']=self.__get_max_class(data_frame)
        return

    if self.__is_terminal(data_frame):
        dictionary_tree['leaf']=data_frame[self.__label_of_target_class][0]
        return
    next_label,updated_data_frame=self.__get_next_node(data_frame,attr_label)
    dictionary_tree[next_label]={}
    data_frames=self.__get_sub_tables(updated_data_frame, next_label)
    for key in data_frames:
        dictionary_tree[next_label][key]={}
        self.__build_tree(data_frames[key],dictionary_tree[next_label][key],next_label,key)
        
  def __complete_tree(self,data_frame,dictionary):
    root=list(dictionary.keys())[0]
    if not isinstance(dictionary[root],dict):
        return
    if len(dictionary[root].keys())!=len(self.__categorized_dictionary_by_target[root].keys()):
        for key in self.__categorized_dictionary_by_target[root]:
            if key not in dictionary[root]:
                dictionary[root][key]={}
                dictionary[root][key]['leaf']=self.__get_max_class(data_frame,self.__label_of_target_class)
    for key in dictionary[root]:
        self.__complete_tree(data_frame,dictionary[root][key], self.__label_of_target_class)


  def __get_max_class(self,data_frame):
        column=data_frame[self.__label_of_target_class]
        histogram=self.__build_histogram(column)
        max_value=self.__get_max_attr(histogram)
        return max_value
            
  def __is_terminal(self,data_frame):
        first_val=data_frame[self.__label_of_target_class][0]
        for val in data_frame[self.__label_of_target_class]:
            if val!=first_val:
                return False
        return True
        
  def __get_sub_tables(self,data_frame,filter_label):
        filter_column=data_frame[filter_label]
        histogram=self.__build_histogram(filter_column)
        list_of_sub_tables={}
        for key in histogram:
            sub_table=data_frame.loc[data_frame[filter_label] == key]
            sub_table=sub_table.loc[:, sub_table.columns != filter_label]
            sub_table=sub_table.reset_index(drop=True)
            list_of_sub_tables[key]=(sub_table)
        return list_of_sub_tables
    
  def __get_next_node(self,data_frame,attr_label):
        updated_date_frame=data_frame.loc[:,data_frame.columns!=attr_label]
        X_data=data_frame.loc[:,updated_date_frame.columns!= self.__label_of_target_class]
        y_data=data_frame[self.__label_of_target_class]
        next_label=self.__calculate_node_with_heighest_gain(X_data,y_data)
        return next_label,updated_date_frame
    
    
  def __calculate_node_with_heighest_gain(self,X_data,y_data):
        entropy_of_s=self.__calculate_entroopy_for_target(y_data)
        categorized_dictionary,categorized_dictionary_by_target=self.__build_categorized_dictionaries(X_data, y_data)
        entropy_dictionaries=self.__build_entropy_dictionaries(categorized_dictionary_by_target)
        gain_dictionaries=self.__build_gain_dictionaries(entropy_dictionaries, categorized_dictionary, entropy_of_s)
        return self.__get_max_attr(gain_dictionaries)
    
    
  def __build_categorized_dictionaries(self,X_data,y_data):
        categorized_dictionary={}
        categorized_dictionary_by_target={}
        for column in X_data:
            categorized_dictionary[column]=collections.defaultdict(int)
            categorized_dictionary_by_target[column]={}
            for i in range(len(y_data)):
                  categorized_dictionary[column][X_data[column][i]] += 1
                  if X_data[column][i] not in categorized_dictionary_by_target[column]:
                      categorized_dictionary_by_target[column][X_data[column][i]]=collections.defaultdict(int)
                  categorized_dictionary_by_target[column][X_data[column][i]][y_data[i]]+=1
        return categorized_dictionary,categorized_dictionary_by_target
    
  def __build_entropy_dictionaries(self,categorized_dictionary_by_target):
        entropy_dictionaries={}
        for first_level_key in categorized_dictionary_by_target:
            entropy_dictionaries[first_level_key]={}
            for second_level_key in categorized_dictionary_by_target[first_level_key]:
                entropy_dictionaries[first_level_key][second_level_key]=self.__calculate_entropy(categorized_dictionary_by_target[first_level_key][second_level_key])
        return entropy_dictionaries
    
  def __get_gain_dictonaries(self,X_data,y_data):
        entropy_of_s=self.__calculate_entroopy_for_target(y_data)
        categorized_dictionary,categorized_dictionary_by_target=self.__build_categorized_dictionaries(X_data, y_data)
        entropy_dictionaries=self.__build_entropy_dictionaries(categorized_dictionary_by_target)
        gain_dictionaries=self.__build_gain_dictionaries(entropy_dictionaries, categorized_dictionary, entropy_of_s,False)
        return gain_dictionaries

    
  def __build_gain_dictionaries(self,entropy_dictionaries,categorized_dictionary,entropy_of_s,use_gain_ratio=True):
        gain_dictionaries={}
        for key in entropy_dictionaries:
            if use_gain_ratio:
                gain_dictionaries[key]=self.__calculate_gain_ratio(entropy_dictionaries[key],categorized_dictionary[key],entropy_of_s)
            else:
                gain_dictionaries[key]=self.__calculate_gain(entropy_dictionaries[key],categorized_dictionary[key],entropy_of_s)
        return gain_dictionaries
    
  def __calculate_gain(self,entropy_dictionary,totals_dictionary,entropy_of_s):
        gain=0;
        total=self.__get_values_from_histogram(totals_dictionary)[1]
        for key in entropy_dictionary:
            gain+=(totals_dictionary[key]/total)*entropy_dictionary[key]
        return entropy_of_s-gain
    
  def __calculate_gain_ratio(self,entropy_dictionary,totals_dictionary,entropy_of_s):
        gain=0;
        split_info=0
        total=self.__get_values_from_histogram(totals_dictionary)[1]
        for key in entropy_dictionary:
            ratio=totals_dictionary[key]/total
            gain+=(totals_dictionary[key]/total)*entropy_dictionary[key]
            split_info+=(-1)*ratio*math.log2(ratio)
        if gain==0 or split_info==0:
            return 0
        information_gain=entropy_of_s-gain
        gain_ratio=information_gain/split_info
        return gain_ratio
    
            
  def __calculate_entroopy_for_target(self,y_data):
        histogram=self.__build_histogram(y_data)
        return self.__calculate_entropy(histogram)
    
  def __calculate_entropy(self,histogram):
        values,total=self.__get_values_from_histogram(histogram)
        entropy=0
        for value in values:
            entropy+=(-1*value/total)*math.log2(value/total)
        return entropy
    
  def __build_histogram(self,column):
        target_dictionary = collections.defaultdict(int)
        for value in column:
            target_dictionary[value] += 1
        return(target_dictionary)
    
  def __get_values_from_histogram(self,input_dictionary):
        values=[]
        total=0
        for key in input_dictionary:
            values.append(input_dictionary[key])
            total+=input_dictionary[key]
        return values,total
    
  def __get_max_attr(self,gain_dictionaries):
        max_attr= max(gain_dictionaries, key=lambda k: gain_dictionaries[k])
        return max_attr
    
  def __get_continues_columns(self,data_frame):
        list_of_continues_columns=[]
        list_of_numric_columns=[]
        for column in data_frame.columns:
            if column==self.__label_of_target_class:
                continue
            if(is_numeric_dtype(data_frame[column])):
                list_of_numric_columns.append(column)
                histogram=self.__build_histogram(data_frame[column])
                if len(histogram)>3 and len(histogram)>len(data_frame[self.__label_of_target_class])/5:
                    list_of_continues_columns.append(column)
        return list_of_continues_columns
    
  def __categorize_continues_columns(self,data_frame,list_of_continues_columns):
        entropy_of_s=self.__calculate_entroopy_for_target(data_frame[self.__label_of_target_class])
        threshold_dctionary={}
        for column in list_of_continues_columns:
            threshold_dctionary[column]=self.__convert_continues_to_descrete_and_return_threeshold(data_frame,column,entropy_of_s)
        return threshold_dctionary
    
  def __convert_continues_to_descrete_and_return_threeshold(self,data_frame,column,entropy_of_s):
        updated_data_frame=data_frame.sort_values(column)
        updated_data_frame=updated_data_frame.reset_index()
        y_data=updated_data_frame[self.__label_of_target_class]
        threeshold=-1
        max_gain=0
        new_column=[]
        for i in range(1,len(updated_data_frame[column])-1):
            if i<len(updated_data_frame[column])-1 and updated_data_frame[column][i]==updated_data_frame[column][i+1]:
                continue
            cell_value=str(updated_data_frame.iloc[i, updated_data_frame.columns.get_loc(column)])
            for j in range(len(updated_data_frame[column])):
                if(j<=i):
                   new_column.append('less_equal_'+cell_value)
                else:
                   new_column.append('greater_'+cell_value)
            X_data=pd.DataFrame (new_column, columns = [column])
            gain=self.__get_gain_dictonaries(X_data, y_data)[column]
            new_column=[]
            if gain>max_gain:
                max_gain=gain
                threeshold=updated_data_frame.iloc[i, updated_data_frame.columns.get_loc(column)];
        for i in range(len(data_frame[column])):
            if data_frame[column][i]>threeshold:
                  data_frame.iloc[i, data_frame.columns.get_loc(column)]='greater_'+str(threeshold)
            else:
                  data_frame.iloc[i, data_frame.columns.get_loc(column)]='less_equal'+str(threeshold)
        return threeshold
    
  def __prepare_data(self,data_frame):
        list_of_continues_columns=self.__get_continues_columns(data_frame)
    
        return self.__categorize_continues_columns(data_frame, list_of_continues_columns)
    
  def __update_testing_data(self,data_frame,threeshold_dictionary):
        for column in threeshold_dictionary:
            threeshold=threeshold_dictionary[column]
            for i in range(len(data_frame[column])):
                 if data_frame.iloc[i, data_frame.columns.get_loc(column)]>threeshold:
                     data_frame.iloc[i, data_frame.columns.get_loc(column)]='greater_'+str(threeshold)
                 else:
                      data_frame.iloc[i, data_frame.columns.get_loc(column)]='less_equal'+str(threeshold)
                    

def load_data():
    df = pd.read_csv('beer.txt', sep="\s",skiprows=1,engine='python',
                      names=['calorific_value','nitrogen','turbidity','style','alcohol','sugars'
                            ,'bitterness','beer_id','colour','degree_of_fermentation'])
    return df,'style'
    


def main():
    df,target_class_label=load_data()
    
    for i in range(10):
          df = df.sample(frac=1).reset_index(drop=True)
          # Start of classification using my class
          df = df.sample(frac=1).reset_index(drop=True)
          training_data = df.iloc[:100,:].reset_index(drop=True)
          testing_data = df.iloc[101:,:].reset_index(drop=True)
          classifier=C45_Tree(training_data, target_class_label)
          classifier.fit()
          correct_predictions_count,predicted_data=classifier.test(testing_data)
          # End of my classification
          
          # Start of classification using Sklear
          training_data = df.iloc[:100,:].reset_index(drop=True)
          testing_data = df.iloc[101:,:].reset_index(drop=True)
          
          X_train=training_data.loc[:,training_data.columns!= target_class_label]
          y_train=training_data[target_class_label]
          
          X_test=testing_data.loc[:,testing_data.columns!= target_class_label]
          y_test=testing_data[target_class_label]
          
          classifier = tree.DecisionTreeClassifier(criterion='entropy')
          classifier.fit(X_train, y_train)
          y_pred = classifier.predict(X_test)
          

          
          print("My Accuracy Score (%)): "+str(int(correct_predictions_count/len(predicted_data)*100))+'%',' --> ',
                 "Sklearn Accuracy Score (%): "+str(int(classification_report(y_test, y_pred,output_dict=True)['accuracy']*100))+'%')
          

          # this will throw an excpetion if the out label is numric. 
          for index in range(len(y_test)):
               if y_pred[index]!=y_test[index]:
                   y_pred[index]='('+str(y_pred[index])+')'
               if predicted_data[index]!=y_test[index]:
                   predicted_data[index]='('+str(predicted_data[index])+')'
          
          
          output_file = pd.DataFrame({'Predicted Data Using Sklear': y_pred,
                                                         'Predicted Data Using My C4.5 Class': predicted_data,
                                                         'Actual Data': y_test})
          output_file.to_csv(index=False,sep='\t')
          output_file.to_csv('attempt_'+str(i+1)+'.csv',sep='\t')
          
              

    

if __name__ == '__main__':
    main()
