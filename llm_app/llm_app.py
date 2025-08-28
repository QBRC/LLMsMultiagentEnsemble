import os
import shutil
import time
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import json
from collections import OrderedDict
import re
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, recall_score, precision_score, jaccard_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import * 

import torch
import fire

from llm_agent import LLMAgent
from fix_json import *


class LLMApp:
    def __init__(self, 
                 config_yaml_file_path:str,
                ):
        """Constructor of LLMApp.

        Keyword arguments:
        config_yaml_file_path -- the path of the LLM application configuration file in YAML.
        """
        with open(config_yaml_file_path, 'r') as f:
            try:
                self.app_config = yaml.safe_load(f)
                # Project
                try:                    
                    self.home_path = os.path.realpath(self.app_config['home_path'])
                except:
                    self.home_path = os.path.realpath('.')
                print(f'LLM agent app home path: {self.home_path}')
                try:                    
                    self.data_folder_path = os.path.join(self.home_path,self.app_config['data_folder_name'])
                except:
                    self.data_folder_path = os.path.join(self.home_path, 'data')                    
                self.llm_output_path = os.path.join(self.data_folder_path, 'llm_output')
                # LLM
                self.model = self.app_config['model']
                self.model_id = self.app_config['model_id']
                self.framework = self.app_config['framework']
                # prompt
                self.sys_msg_file_path = os.path.realpath(self.app_config['system_message_file_path'])
                self.prompt_file_path = os.path.realpath(self.app_config['prompt_file_path'])
                self.prompt_id = self.app_config['prompt_id']
                # input data
                self.input_data_path = os.path.realpath(self.app_config['input_data_path'])
                self.dataset_id = self.app_config['dataset_id']
                self.predictor = f'{self.model_id}_{self.prompt_id}'
                self.input_text_col = self.app_config['input_text_column_name']
                self.input_data_id_col = self.app_config['input_data_id_column_name']
                # output data
                self.var_val_dict_path = os.path.realpath(self.app_config['var_val_dict_path'])
                self.json_output_template_path = os.path.realpath(self.app_config['json_output_template_path'])
                try:
                    with open(self.json_output_template_path, 'r') as f:
                        json_template = json.load(f,object_pairs_hook=OrderedDict) # text structured, but not verified json
                        self.json_key_list = json_template.keys()
                except:
                    print(f"Error: failed to get keys from {self.app_config['json_output_data_template']}\n{e}")
            except yaml.YAMLError as e:
                print(f"Error: reading yaml configuration file failed. {e}")
        

    def compute(self):
        """Set up and use LLM to compute."""

        # Create a LLM agent.
        agent = LLMAgent(framework=self.framework, model=self.model)
        
        # Read in input data.
        df = pd.read_csv(self.input_data_path)
        
        # Ensure the input text col is text type
        df[self.input_text_col] = df[self.input_text_col].astype(str)
        
        # Read in prompt.
        with open(self.sys_msg_file_path, 'r') as f:
            sys_msg = f.read()
        with open(self.prompt_file_path, 'r') as f:
            instruction = f.read()
        with open(self.var_val_dict_path, 'r') as f:
            var_val_list = f.read() # text structured, but not verified json
        with open(self.json_output_template_path, 'r') as f:
            json_template = f.read() # text structured, but not verified json
            # json_template = json.load(f) # text has no structure, but verified json format

        instruction = instruction.replace('{{<Variable-Value List>}}',var_val_list)   
        instruction = instruction.replace('{{<json_output_template>}}',json_template)   

        # Create LLM output path, if not exist.
        os.makedirs(self.llm_output_path, exist_ok=True)
        
        # Save system_message and prompt to LLM output folder, being a part of provenance.
        with open(os.path.join(self.llm_output_path, 'system_message.txt'), 'w') as f:
            f.write(sys_msg)
        with open(os.path.join(self.llm_output_path, 'prompt.txt'), 'w') as f:
            f.write(instruction)
        # with open(os.path.join(self.llm_output_path, 'prompt.yaml'), 'w') as f:
        #     yaml.dump([{"system_message": [sys_msg], "prompt": [instruction]}], f)
            
        # Create cols to save the values to be generated by LLM in df
        for col in self.json_key_list:
            df[col] = None

        # Make batch call to LLM.
        df = self.make_batch_call(
                agent=agent, 
                df=df,
                system_message=sys_msg,
                instruction=instruction, 
                prompt_id=self.prompt_id,
                dataset_id=self.dataset_id,
                key_list=self.json_key_list,
                model_id=self.model_id, 
                output_path=self.llm_output_path,
                )
        # Save output to a csv file.
        csv_file_path = os.path.join(self.data_folder_path, f"{self.dataset_id}_{self.model_id}_{self.prompt_id}.csv")
        df.to_csv(csv_file_path, index=False)

        print(f"\nThe output csv file saved to {csv_file_path}.")
        print(f"Computing with {self.model_id} for {self.dataset_id} succeed.\n")


    def make_batch_call(self,
            agent:LLMAgent,
            df:pd.DataFrame, 
            output_path:str='./data/', 
            instruction:str='hi', 
            system_message:str='', 
            key_list:list=[], # expected key_list of json data retuned from GPT; same as cols_to_fill in df
            dataset_id:str='',
            model_id:str='llama3.1', # default model
            prompt_id:str='', 
            ):

        print(f"Model: {model_id}")
        print(f"Dataset: {dataset_id}")
        failed_requests = []
        failed_json_loading = []
        start_time = time.time()
        start_time_str = str(datetime.now())
        print(f"Start at: {start_time_str}")  
    
        k = 0
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        temperature = 0.0
        
        for index, row in df.iterrows():
            case_id = df[self.input_data_id_col][index]
            print(f"\r{str(index)}:  {case_id}  ", end='', flush=True)
            report = df.at[index, self.input_text_col]
            
            # replace '\n' as '  '
            report = report.replace('\n', '  ')
                    
            if report == '':
                print(f"Case {case_id} text is empty!")
            else:
                # custruct prompt
                prompt = instruction + report
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],

                # Call the llm -----------------------------------------------------
                response = agent.request(prompt, system_message, temperature=temperature)
                # ------------------------------------------------------------------
                
                if response is None:
                    failed_requests.append(str(case_id))
                    print(f"Warning: Request for {case_id} is failed!\n")
                else:
                    response_content = agent.get_response_content(response)

                    # if the response_content is '' or None
                    if not response_content: 
                        failed_requests.append(str(case_id))
                        print(f"Warning: Failed to get the response content for {case_id}.")
                    else:
                        # write the original response_content to a text file in destination folder
                        filename = f"{case_id}_{model_id}_{prompt_id}.json"
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        with open(os.path.join(output_path, filename), 'w') as f2:
                            f2.write(response_content)
                            
                        # Extract record from response content. -----------------------------
                        record = self.extract_record_from_llm_output(response_content, key_list)    
                        # -------------------------------------------------------------------
                        if (record is not None) and len(key_list) > 0:
                            for col in key_list:
                                df.at[index, col] = record[col]
                        else:
                            failed_json_loading.append(str(case_id))
                
                    # Count tokens consumed
                    input_tokens_i, output_tokens_i, total_tokens_i = agent.get_tokens_consumed(response)
                    input_tokens += input_tokens_i if isinstance(input_tokens_i, int) else 0
                    output_tokens += output_tokens_i if isinstance(output_tokens_i, int) else 0
                    total_tokens += total_tokens_i if isinstance(total_tokens_i, int) else 0

        end_time = time.time()
        elapsed_time = end_time - start_time
        end_time_str = str(datetime.now())
        print(f"\nEnd at: {end_time_str}")        
        print(f"Elapsed time: {elapsed_time/60:2f} minutes ({elapsed_time:2f} seconds).\n")
        print(f"Failed requests: {str(failed_requests)}")
        print(f"Failed json loading: {len(failed_json_loading)}\n{str(failed_json_loading)}")
        
        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")
        print(f"Total Tokens: {total_tokens}")

        # Estimate cost
        estimated_cost = self.estimate_cost(model=self.model, input_tokens=input_tokens,  output_tokens=output_tokens)
        if estimated_cost > 0:
            print(f"Estimated cost: ${estimated_cost}")
    
        # # save the llm output 
        # csv_file_path = f"{output_path}{dataset_id}_{model_id}_{prompt_id}_raw.csv"
        # df.to_csv(csv_file_path, index=False)
    
        # copy json files with error to folder 'json_with_errors'
        if failed_json_loading:
            json_with_errors_path = os.path.join(self.data_folder_path, 'json_with_errors')
            os.makedirs(json_with_errors_path, exist_ok=True)
            for record in failed_json_loading:
                json_file_path = os.path.join(output_path, f'{record}_{self.predictor}.json')
                json_file_w_errs_path = os.path.join(json_with_errors_path, f'{record}_{self.predictor}.json')
                shutil.copy(json_file_path, json_file_w_errs_path)
        
        # Save provenance to provenance.yaml
        provenance = [{
            "time_started": start_time_str,
            "time_ended": end_time_str,
            "time_used": f"{elapsed_time/60:2f} minutes",
            "model": self.model,
            "framework": self.framework,
            # "system_message": system_message,
            # "prompt": instruction,
            "dataset_id": dataset_id,
            "input_data_path": self.input_data_path,
            "temperature": temperature,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": f"${estimated_cost}",
            "failed_request": failed_requests,
            "number_of_failed_json_loading": len(failed_json_loading),
            "failed_json_loading": failed_json_loading,
            }]
        with open(os.path.join(output_path, 'provenance.yaml'), 'w') as f:
            yaml.dump(provenance, f)
    
        return df


    def extract_record_from_llm_output(self, 
                                       response_content:str, 
                                       key_list:str,
                                      ):
        """Extract a record from LLM response."""
        
        record = None
    
        try:
            json_data = json.loads(response_content)
            if json_data:
                # Create a new record from key_list and flattened_data:
                record = {key: str(json_data.get(key, '')).strip() for key in key_list}
                # print(f"Extracted record:{record}")
        except Exception as e:     
            # Failed to load the response_content as json data;
            print(f"Parsing JSON failed. Error: {e}")
            print(f"Trying to fix json error:")
            # call helper function fix_json
            json_str = fix_json(json_string=response_content, verbose=True)
            if json_str:
                json_data = json.loads(json_str)
                record = {key: str(json_data.get(key, '')).strip() for key in key_list}
               
        return record


    def estimate_cost(self, model, input_tokens, output_tokens):
        if model in ['GPT-4-Turbo', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt4t1106', 'GPT-4-Turbo-Vision']:
            cost = input_tokens * 0.001 * 0.01 + output_tokens * 0.001 * 0.03
        elif model in ['GPT-4o', 'gpt-4o']: #2024-05-23 openAI pricing
            cost = input_tokens * 0.000001 * 5 + output_tokens * 0.000001 * 15
        elif model in ['GPT-4o-mini', 'gpt-4o-mini']: #2024-07-26 openAI pricing
            cost = input_tokens * 0.000001 * 0.15 + output_tokens * 0.000001 * 0.06
        elif model in ['GPT-4.1-mini', 'gpt-4.1-mini']: #2025-08-21 openAI pricing
            cost = input_tokens * 0.000001 * 0.4 + output_tokens * 0.000001 * 1.60
        elif model in ['GPT-4-32']:
            cost = input_tokens * 0.001 * 0.06 + output_tokens * 0.001 * 0.12
        elif model in ['gpt-4-0613']: #8k
            cost = input_tokens * 0.001 * 0.03 + output_tokens * 0.001 * 0.06
        elif model in ['GPT-3.5-Turbo-0125', 'gpt-35-turbo-0125']: #16k
            cost = input_tokens * 0.001 * 0.0005 + output_tokens * 0.001 * 0.0015
        elif model in ['gpt-3.5-turbo-1106','gpt35t16k1106', 'gpt35t1106']: #16k
            cost = input_tokens * 0.001 * 0.0015 + output_tokens * 0.001 * 0.002
        elif model in ['gpt-35-turbo-16k']:
            cost = input_tokens * 0.001 * 0.003 + output_tokens * 0.001 * 0.004
        elif model in ['GPT-3.5-Turbo-Instruct']: #4k
            cost = input_tokens * 0.001 * 0.0015 + output_tokens * 0.001 * 0.002
        else:
            print(f"Model {model} is not in the cost list.")
            cost = 0
        return cost



    def load_llm_output_from_json_files(self,
                                        config_yaml_file_path:str='./config.yaml',
                                        output_path:str='./data/',
                                       ):
        """ Load llm output json files from a folder

        Note: 
        - Condition: the json files already generated from LLM and saved in llm_output folder.
        - The configuration yaml file could be the same one as the LLM application has, and could be different one, 
        but should contain the following keys:
            input_data_path, (the input csv file used for LLM to generate json data)
            dataset_id,
            input_data_id_column_name
            model_id,
            prompt_id,
            json_key_list
        """
        if config_yaml_file_path:
            with open(config_yaml_file_path, 'r') as f:
                app_config = yaml.safe_load(f)
                json_data_folder_path = self.llm_output_path
                input_data_path = app_config['input_data_path']
                id_col_name = app_config['input_data_id_column_name']
                dataset_id = app_config['dataset_id']
                model_id = app_config['model_id']
                prompt_id = app_config['prompt_id']
                key_list = app_config['json_key_list'] 
                
                # Read in original input dataset into df
                df = pd.read_csv(input_data_path)
        
                # Initialize columns produced from LLM
                for key in key_list:
                    df[key] = ''
            
                # for each row, load json data from file
                for index, row in df.iterrows():
                    case_id = df[id_col_name][index]
                    print(f"\r{str(index)}:  {case_id}  ", end='', flush=True)
                    json_file_path = os.path.join(json_data_folder_path, f'{case_id}_{model_id}_{prompt_id}.json')
                    try:
                        with open(json_file_path, 'r') as json_file:
                            response_content = json_file.read()
                            record = self.extract_record_from_llm_output(response_content, key_list)
                        for col in key_list:
                            df.at[index, col] = record[col]
                    except Exception as e:
                        print(f"Error: {e}")
                
                # save the loaded json data
                csv_file_path = os.path.join(self.data_folder_path, f"{self.dataset_id}_{self.model_id}_{self.prompt_id}.csv")
                df.to_csv(csv_file_path, index=False)
                print(f"\nLoaded the json data in {json_data_folder_path} and merged into {input_data_path}.\nSave to {csv_file_path}.")
        
                return df
        else:
            print("Please provide a valid configuration yaml file path for parameter config_yaml_file_path.")
            return None
                  

    def evaluate_with_cm(self,reference,prediction,title,predictor):
    
        df_metrics = pd.DataFrame()
        
        if len(reference) == len(prediction):   
            # plot confusion matrix and calculate metrics of interest
            df_metrics = self.plot_cm_with_metrics( 
                reference=reference, 
                prediction=prediction, 
                title=title,
                predictor=predictor
                )
        else:
            print(f"Error: prediction length ({len(pred)}) is not equal to reference length ({len(ref)}.")
            
        return df_metrics 
    
    
    
    def plot_cm_with_metrics(self,
                             reference, 
                             prediction, 
                             title, 
                             predictor, 
                             labels=None, 
                             save_plots_to_path=None, 
                             df_metrics=None,
                            ):
    
        # plot confusion matrix
        self.plot_cm(
            reference = reference,
            prediction = prediction,
            title = title,
            predictor = predictor,
            labels = labels,
            save_plots_to_path = save_plots_to_path,
        )
        
        # calulate metrics
        df_mtr = self.calculate_metrics(
                     reference, 
                     prediction, 
                     predictor, 
                     df_metrics,
                     verbos=True
                    )
        return df_mtr
    
    
    def calculate_metrics(self,
                 reference, 
                 prediction, 
                 predictor, 
                 df_metrics=None,
                 verbos=False):    
        
        # calculate metrics 
        accuracy = accuracy_score(reference, prediction)
        f1 = f1_score(reference, prediction, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(reference, prediction)
    
        recall_w = recall_score(reference, prediction, average='weighted',zero_division=0) 
        recall = recall_score(reference, prediction, average=None, zero_division=0)
    
        precision_w = precision_score(reference, prediction, average='weighted',zero_division=0)
        precision = precision_score(reference, prediction, average=None, zero_division=0)
    
        num_cases = len(prediction)
    
    
        if verbos:
            d=3
            print(f"Accuracy: {round(accuracy,d)}")
            print(f"F1 (weighted): {round(f1,d)}")
            print(f"Kappa: {round(kappa,d)}")
            print(f"Recall (weighted): {round(recall_w,d)}")    
            print(f"Precision (weighted ): {round(precision_w,d)}")
            print(f"Number of Cases: {num_cases}")
            
        columns = ['Predictor','Accuracy', 'F1', 'Recall (weighted)', 'Precision (weighted)', 'Cases']
        df_row = pd.DataFrame([[predictor, accuracy, f1, recall_w, precision_w, num_cases]], columns=columns)
        
        if df_metrics:
            if df_metrics.columns == df_row.columns:
                df_metrics = pd.concat([df_metrics, df_row], ignore_index=True)
                return df_metrics
            else:
                print(f"Warning: df_metrics and calculated metrics have different columns.\n Retune the caculated metrics only.")
                return df_row
        else:
            return df_row
    
    def plot_cm(self,
                reference, 
                prediction, 
                title, 
                predictor, 
                labels = None, # Specify labels in an intended order; None: use default order 
                save_plots_to_path = None, # specify folder to save plot; None: not save
               ):
    
        if labels is None:
            labels = list(set(reference.astype(str)) | set(prediction.astype(str)))
            labels.sort()
            labels = self.move_items_to_list_end(labels, ['Other', 'Not specified', 'Cannot be determined'])
        else:
            try:
                if set(labels) != set(reference).uniion(set(prediction)):
                    raise Error('Parameter labels do not match the reference and prediction!')
            except Error as e:
                print(f"Error: {e}")
        # print(f" Number of labels: {len(labels)}")
        
        # Calculate the confusion matrix
        cm = confusion_matrix(reference, prediction, labels = labels)
    
        # Visualize the confusion matrix using a heatmap
        fig_size = (lambda x: 3 if x <=2 else (x if x > 2 and x <= 20 else 20))(len(labels))
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=True, xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel(f"Prediction")
        plt.ylabel("Reference")
        fig = plt.gcf()
        plt.show()
        if save_plots_to_path:
            filepath = os.path.join(
                save_plots_to_path,
                title+'.pdf')                   
            fig.savefig(filepath, format='pdf')
        plt.close()
    
    def move_items_to_list_end(self, list, end_item_list):
        for item in end_item_list:
            if item in list:
                index = list.index(item)
                list.pop(index)
                list.append(item)
        return list


def aggregated_multilabel_cm(y_true, y_pred, title):
    """Aggregate multilabel confusion matrixes and calculate metrics."""
    
    # create 
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    micro_TN = sum(cm[:,0, 0])
    micro_FN = sum(cm[:,1, 0])
    micro_TP = sum(cm[:,1, 1])
    micro_FP = sum(cm[:,0, 1])
    
    labels = ['N','P']
    
    cm_aggr = pd.DataFrame([[micro_TN,micro_FP],[micro_FN,micro_TP]], index=labels, columns=labels)
    # print(cm_aggr)
    # print("N", micro_TN,micro_FP)    
    # print("P", micro_FN,micro_TP)  

    # Visualize the confusion matrix using a heatmap
    fig_size = (lambda x: 3 if x <=2 else (x if x > 2 and x <= 20 else 20))(len(labels))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm_aggr, annot=True, cmap='Blues', fmt='d', cbar=True, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("Reference")
    fig = plt.gcf()
    plt.show()
    
    micro_recall = micro_TP/(micro_TP + micro_FN)
    micro_prec = micro_TP/(micro_TP + micro_FP)
    micro_accuracy = (micro_TP + micro_TN)/(micro_TN + micro_FN + micro_TP + micro_FP)
    print(f"Accuracy (micro-average): {round(micro_accuracy,3)}")
    print(f"Recall (micro-average): {round(micro_recall, 3)}")
    print(f"Precision (micro-average): {round(micro_prec,3)}")
    

def run():
    """Run the LLM application with the config.yaml to generate output for input data."""
    app = LLMApp('config_llm_app.yaml')
    app.compute()
    # app.load_llm_output_from_json_files()
    # to evaluate result, use evaluate_with_cm() in a jupyter notebook to see the plots. 
    

# Run the LLM application in commandline
if __name__ == "__main__":
    torch.cuda.empty_cache() 
    # run LLM application configured by config.yaml 
    fire.Fire(run)

