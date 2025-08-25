import os
import sys
import time
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import json
from collections import OrderedDict
import math
import re
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import *
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, recall_score, precision_score,jaccard_score
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import fire
from abc import ABC, abstractmethod 

from llm_agent import LLMAgent
from llm_app import LLMApp
from fix_json import *

# define Abstract Base Class as interface for user defined postprocessing functions
class Postprocessing(ABC):
    @abstractmethod
    def excute(self,df:pd.DataFrame) -> pd.DataFrame:
        pass

class EnsembleLLMsApp:
    def __init__(self, 
                 config_yaml_file_path):
        """Constructor of LLMApp.

        Keyword arguments:
        config_yaml_file_path -- the path of the LLM application configuration file in YAML.
        """
        with open(config_yaml_file_path, 'r') as f:
            try:
                self.app_config = yaml.safe_load(f)
                
                # Project
                self.project_name = self.app_config['project_name']
                self.home_path = os.path.realpath('.')
                print(f'Home path: {self.home_path}')
                try:                    
                    self.data_folder_path = os.path.join(self.home_path, os.path.realpath(self.app_config['data_folder_name']))
                except:
                    self.data_folder_path = os.path.join(self.home_path, 'data')   
                
                # Prompt
                self.prompt_file_path = os.path.join(self.home_path,os.path.realpath(self.app_config['prompt_file_path']))
                self.prompt_id = self.app_config['prompt_id']
                try:
                    self.sys_msg_file_path = os.path.join(self.home_path,os.path.realpath(self.app_config['system_message_file_path']))
                except:
                    self.sys_msg_file_path = None
                
                # Input data
                self.input_data_path = os.path.join(self.home_path,os.path.realpath(self.app_config['input_data_path']))
                self.dataset_id = self.app_config['dataset_id']
                self.input_text_col = self.app_config['input_text_column_name']
                self.input_data_id_col = self.app_config['input_data_id_column_name']

                # Output data
                self.var_val_list_path = os.path.join(self.home_path,os.path.realpath(self.app_config['var_val_list_path']))
                try:
                    with open(self.var_val_list_path, 'r') as f:
                        self.var_val_list = json.load(f,object_pairs_hook=OrderedDict) # keep original order
                except Exception as e:
                    print(f"Error: failed to get variable-value list from {self.var_val_list_path}\n{e}")

                self.json_output_template_path = os.path.join(self.home_path,os.path.realpath(self.app_config['json_output_template_path']))
                try:
                    with open(self.json_output_template_path, 'r') as f:
                        json_template = json.load(f,object_pairs_hook=OrderedDict) # keep original order
                        self.json_key_list = list(json_template.keys())
                except Exception as e:
                    print(f"Error: failed to get keys from {self.json_output_template_path}\n{e}")
                # verify vars in json output template
                for v in self.var_val_list.keys():
                    if v not in self.json_key_list:
                        print(f"Error: Variable {v} is not specified in the JSON output template.")
                        sys.exit()

                # Specs for LLM agent apps
                self.llm_apps_specs = self.app_config['llm_apps']
                  
            except yaml.YAMLError as e:
                print(f"Error: reading ensemble configuration yaml file failed.\n{e}")
                sys.exit()

        if len(self.llm_apps_specs)==0: 
            print("Error: llm_apps must have at least one LLM Agent app specified. ")
        else: 
            # Initialize the list for saving LLMAgent objects
            self.llm_apps = []

            # create a dictionary containing common key-value pairs to all llm_apps
            dic = {
                'system_message_file_path': self.sys_msg_file_path,
                'prompt_file_path': self.prompt_file_path,
                'prompt_id': self.prompt_id,
                'input_data_path': self.input_data_path,
                'dataset_id': self.dataset_id,
                'input_data_id_column_name': self.input_data_id_col,
                'input_text_column_name': self.input_text_col,
                'var_val_list_path': self.var_val_list_path,
                # 'var_val_list': self.var_val_list,
                'json_output_template_path':self.json_output_template_path,
                'json_key_list': self.json_key_list
            }
            
            # create full configuration for each LLM Agent app
            predictors = []
            for app in self.llm_apps_specs:
                predictor  = f"{app['model_id']}_{self.prompt_id}"
                predictors += [predictor]

                # add predictor to the llm_app
                app.update({'predictor': predictor})
                
                # add app_path for the llm_app
                app.update({'home_path': os.path.join(self.home_path, f"{app['predictor']}")})

                # add common key-value pairs to each llm_app
                app.update(dic)
                
                # create folder for each llm_app
                os.makedirs(app['home_path'], exist_ok=True)

                config_yaml_file_path = os.path.join(app['home_path'], 'config_llm_app.yaml')
                # save the config path to app
                app.update({'app_config_file_path':config_yaml_file_path})

                # dump each llm_app's config yaml file to that folder
                with open(config_yaml_file_path, 'w') as f:
                    yaml.dump(app, f)
                    
            self.predictors = predictors        
                    

    def launch_llm_apps(self):
        """Set up and use LLM to compute."""

        # Start to count time.
        start_time = time.time()
        start_time_str = str(datetime.now())
        print(f"Start at: {start_time_str}")  

        # for each llm_agent app specification, create a llm_agent instance
        for app in self.llm_apps_specs: 
            # create LLMApp object for each llm_app specified with config_llm_app.yaml
            llm_app = LLMApp(app['app_config_file_path'])
            # add the created llm_app object to llm_apps list
            self.llm_apps.append(app)
            # launch the created llm_app to serve its application
            llm_app.compute()
                        
        # End time counting
        end_time = time.time()
        elapsed_time = end_time - start_time
        end_time_str = str(datetime.now())
        print(f"\nEnd at: {end_time_str}")        
        print(f"Elapsed time: {elapsed_time/60:2f} minutes ({elapsed_time:2f} seconds).\n")

    
    def aggregate(self):
        """Aggregate the output csv files from every llm_app."""

        # Read in input data as base data.
        df = pd.read_csv(self.input_data_path)
        
        # Add keys in json_key_list to the df as columns and initialize to '', 
        # for adding postfix f"_{predictor}" to those cols, when merging each llm app's prediction.
        for col in self.json_key_list:
            df[col] = ''

        # Read in and merge each llm app's prediction
        for app in self.llm_apps_specs:
            predictor = f"{app['model_id']}_{self.prompt_id}"
            # Read in app's output.
            print(f"\nRead in prediction from {app['home_path']}/data/")
            df1 = pd.read_csv(os.path.join(app['home_path'], f"data/{app['dataset_id']}_{predictor}.csv"),
                             keep_default_na=False)
            # Select cols to merge to main df.
            df1 = df1[[self.input_data_id_col] + self.json_key_list]
            # merge        
            df = pd.merge(df, df1, on=self.input_data_id_col, how='left', suffixes=('', f"_{predictor}"))

        # Drop cols in json_key_list, after merging
        df.drop(columns=self.json_key_list, inplace=True)
        
        # Save the aggregated file to data folder
        aggregated_file_path = os.path.join(self.data_folder_path, f"{self.dataset_id}_{self.project_name}_aggregated.csv")
        df.to_csv(aggregated_file_path, index=False)
        print(f"\nSave the aggregated data into {os.path.relpath(aggregated_file_path,self.home_path)}")
        return df

    
    def vote(self,df:pd.DataFrame,var_val_list:str='',min_winning_share:list=[]):
        """Vote function:
        For each variable in vars predicted by each of predictors, use maximum vote above threshold method to vote, 
        e.g. 7 voters vote for 3 options, the option with at least 3 votes win (threshold 3/7)
        return vote result in df.

        df: input df contains the predictions for a list of variables by predictors
        vars: a list of categorical variables, to be vote for.
        predictors: a set of predictors
        ensemble_name: the name of the LLMs multiagent ensemble
        min_winning_share: a list of minimal vote rate to win for each variable; default: empty, use natural minimal for each var.
        """
        
        print(f"Voting by predictors: {self.predictors}")
        
        # Vote for each variable
        for var in self.var_val_list.keys():
            vals = self.var_val_list[f'{var}']
            if len(vals) <= 1:
                print(f"Warning: Variable {var} does not have a list of categorical values; skipped!!\nCurrent value: {vals}.")
            else:
                # get cols where var predicted by each predictor 
                cols_to_check = [
                    f"{var}_{p}"
                    for p in self.predictors
                    if f"{var}_{p}" in df.columns
                ]
                votes_cols = []
                for val in vals: # count votes for each pair of var-val
                    df[f'votes_{var}_{val}'] = 0 # initialize as 0
                    df[f"votes_{var}_{val}"] = (df[cols_to_check] == val).sum(axis=1)
                    votes_cols += [f'votes_{var}_{val}']
                # Handle non-standard values
                df[f'votes_{var}_non-standard'] = 0 # initialize as 0
                df[f"votes_{var}_non-standard"] = (~df[cols_to_check].isin(vals)).sum(axis=1)
                # Handle empty value (LLM failed to respond)
                df[f'votes_{var}_no_response'] = 0 # initialize as 0
                df[f"votes_{var}_non_response"] = (df[cols_to_check] == '').sum(axis=1)

                # Infer var's value with max votes
                df[f'{var}_voted'] = '' # initialize the col <var> for the voted value
                # for each row in df, infer a value
                df[f'{var}_voted'] = df[votes_cols].apply(lambda row: self.infer_value_with_vote(row,vals),axis=1)
        
        vote_result_file_path = os.path.join(self.data_folder_path, f"{self.dataset_id}_{self.project_name}_voted.csv")        
        # vote_result_file_path = os.path.join(self.data_folder_path, f"{self.project_name}_voted.csv")
                                                
        df.to_csv(vote_result_file_path, index=False)
        print(f"Vote result saved to {os.path.relpath(vote_result_file_path,self.home_path)}.")
        return df
        

    def infer_value_with_vote(self,row:pd.Series,vals:list,min_winning_share:float=None,min_num_votes:int=2):
        
        total_votes = row.sum()
            
        if total_votes < min_num_votes: # if total number of valid votes less than min_num_votes, "Review"
            return "Review"
        else:
            votes_share_list = [v/total_votes for v in row]
            max_share = max(votes_share_list)

            if min_winning_share is None: # if no minimum winning share is specified, use simple majority
                min_winning_share = (math.floor(total_votes/2)+1)/total_votes
            
            if max_share < min_winning_share:
                return "Review"
            elif votes_share_list.count(max_share) > 1: # Tie: multiple candidates ties in the max votes share.
                return "Review"
            else: # greater than wining threshold and no tie, return the wining value.
                return vals[votes_share_list.index(max_share)]
                
    def apply_Postprocessing(self, function: Postprocessing) -> pd.DataFrame:
        """
        Interface: Apply any function that implements Postprocessing interface
       
        Args:
            funtion: An instance of a class that implements Postprocessing
       
        Returns:
            Resulting DataFrame after applying the funtion
        """
        try:
            result = funtion.execute(self,**kwargs)
            # to do ...
            return result
        except Exception as e:
            print(f"Error: when applying user defined funtion: {e}")
            return pd.DataFrame()


def run(command:str='',**kwargs) -> EnsembleLLMsApp:
    """Run the ensemble LLM application with the ensemble_config.yaml and launch all the specified llm_apps."""
    
    commands = ['init','launch','aggregate','vote',]
        
    if command in commands:
        try:
            ensemble = EnsembleLLMsApp('./ensemble_config.yaml')
        except Exception as e:
            print(f"Failed to init the EnsembleLLMsApp.\n{e}")
    
    if command=='launch':
        ensemble.launch_llm_apps()
    elif command=='aggregate':
        ensemble.aggregate()
    elif command=='vote':
        df = pd.read_csv(os.path.join(ensemble.data_folder_path, f"{ensemble.dataset_id}_{ensemble.project_name}_aggregated.csv"), 
                         keep_default_na=False)
        if 'var_val_list' in kwargs:
            ensemble.var_val_list = json.loads(kwargs['var_val_list'])
            print()
            print(kwargs['var_val_list'])
            print(ensemble.var_val_list)
            print()
        ensemble.vote(df=df)
    elif command=='evaluate':
        ensemble.evaluate()
    elif command=='' or command=='all':
        ensemble.launch_llm_apps()
        ensemble.aggregate()
        df = pd.read_csv(os.path.join(ensemble.data_folder_path, f"{ensemble.dataset_id}_{ensemble.project_name}_aggregated.csv"), 
                         keep_default_na=False)        
        ensemble.vote(df=df)
    elif command=='init':
        pass
    else:
        print(f"Unrecognized command: {command} \nSupported commands:")
        for cmd in commands:
            print(cmd)
        return None
        
    return ensemble

# Run the LLM application in commandline
if __name__ == "__main__":
    torch.cuda.empty_cache() 
    # run LLM application configured by config.yaml 
    fire.Fire(run)


