import os

from openai import OpenAI
from openai import AzureOpenAI
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
import torch


class LLMAgent:
    # Constructor to initialize the LLM Agent instance    
    def __init__(self, 
                 framework:str='Ollama', # 'Ollama', 'OpenAI', 'AzureOpenAI', or empty (huggingface)
                 model:str='llama3.3', # language model to use, llama3.1:8b as default; or huggingface model path
                 ollama_port = '11434', # default value from Ollama
                 quantification = None # default: torch_dtype=torch.float16
                 ):
        """Constructor to initialize the LLM Agent instance
        
        Keyword arguments:
        framework -- the framework used to call a LLM, e.g. Ollama, OpenAI, Azure OpenAI, or empty (directly use transformers)
        model -- the name of the LLM to use. Naming depends on framework; for AzureOpenAI, use deployment name.

        Notes:
        To use Ollama, download (https://github.com/ollama/ollama) and set up ollama environment first.
        To use Azure OpenAI, set up Azure OpenAI at https://portal.azure.com, and deploy the model to use first.
        To use OpenAI, create OpenAI account first.
        """
        
        self.framework = framework
        self.model = model  
        self.client = None
        self.tokenizer = None

        # ------------------------
        # if framework is empty, use transfomers to construct a LM from model path (huggingface or local)
        if not framework or framework in ['huggingface', 'Huggingface']: 
            # self.tokenizer = AutoTokenizer.from_pretrained(model)
            
            bnb_bits = quantification # config, if use quantification 
            if bnb_bits is None or bnb_bits not in [4,8]:
                self.client = AutoModelForCausalLM.from_pretrained(model, 
                                                             torch_dtype=torch.float16, 
                                                             # quantization_config=quantization_config,
                                                             # low_cpu_mem_usage=True, 
                                                             device_map="auto")
            else:
                # update with info at https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
                if bnb_bits==4:
                    # quantization_config=BitsAndBytesConfig(load_in_4bit=True)
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                elif bnb_bits==8:
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
                    # quantization_config=BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
                else:
                    quantization_config=None
                if quantization_config is not None:
                    self.client = AutoModelForCausalLM.from_pretrained(model, 
                                                                 torch_dtype="auto", 
                                                                 quantization_config=quantization_config,
                                                                 # low_cpu_mem_usage=True, 
                                                                 device_map="auto")

            self.tokenizer = AutoTokenizer.from_pretrained(model)

        # ------------------------
        elif self.framework in ['ollama', 'Ollama']:
            # option 1: use OpenAI API
            self.client = OpenAI(
                # base_url=f"http://localhost:11333/v1/",
                base_url=f"http://localhost:{ollama_port}/v1/",
                # required but ignored
                api_key='ollama',
            )
            # # option 2: use ollama python API 
            # self.client = ollama.Client(host=f'http://localhost:{ollama_port}')
            # # option 3: call ollama.chat(...) directly without explicitly create client
            # pass
        elif self.framework in ['AzureOpenAI', 'Azure OpenAI', 'Azure_OpenAI', 'Azure-OpenAI']:
            self.client = AzureOpenAI(
                api_key = os.getenv("AZURE_OPENAI_KEY"),  
                api_version="2024-10-01-preview",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        elif self.framework in ['openAI', 'OpenAI']:
            # Need to add your OpenAI API Key in environment variable OPENAI_API_KEY 
            self.client = OpenAI()
        else:
            print(f"LLMAgent: the specified 'framework' {framework} is not recognized.")                       
    

    # Function to a request to the language model and get a response
    def request(self, prompt, system_message='', temperature=0.0, json_output=True):
        """Make request to LLM."""

        response = None
        
        # Request via transformers to a LLM in haggingface ----------------------------------
        # Change the follow block as needed for the specific LLM to call
        if (not self.framework) or (self.framework in ['huggingface', 'Huggingface']):
            try:
                # construct messages
                message =  f"{system_message}### User: {prompt}\n\n### Output:\n"
                # tokenize input
                inputs = self.tokenizer(message, return_tensors="pt").to("cuda")
                # get output from LLM with the inputs
                response = self.model.generate(**inputs, 
                                        do_sample=True,
                                        temperature=temperature,
                                        # top_p=0.95, top_k=0, 
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        max_new_tokens=1024)
            except Exception as e:
              print('Error:', e)
        # Request via Ollama ------------------------------------------------------------
        elif self.framework in ['ollama', 'Ollama']:
            try:
                # # Call LLM using ollama python API
                # response = ollma.chat(
                #     model=self.model,
                #     messages=[
                #         {"role": "system", "content": system_message},
                #         {"role": "user", "content": prompt},
                #     ],                    
                # )
                # Call LLM using OpenAI API
                if json_output:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        # max_completion_tokens = 8192, # set on 2025-02-13
                        # comment this line, if no need to output as json data
                        response_format={ "type": "json_object" }, 
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                    )
                    
            except ollama.ResponseError as e:
              print('Error:', e)
        # Request to OpenAI or Azure OpenAI ------------------------------------------------------------------
        elif self.framework in ['AzureOpenAI', 'Azure OpenAI', 'Azure_OpenAI', 'Azure-OpenAI', 'OpenAI', 'openAI']:
            try:
                # call gpt model 
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    # comment this line, if no need to output as json data
                    response_format={ "type": "json_object" }, 
                )
            except Exception as e:
                print(f"Error: the request to GPT {self.model} is failed. \nAn exception of type {type(e).__name__} occurred: {e}")
            
        # Return the content of the response
        return response


    def get_response_content(self, response):
        """Get the text content generated by LLM from the response structure."""

        # Initialize 
        response_content = None
        
        if response:
            if not self.framework: 
                # get response contents from LLM agent using transformers
                response_content = self.tokenizer.decode(response[0], skip_special_tokens=True)
                response_content = self.extract_answer(response_content)
                # how to get token usage generally? -- temporarily set to 0.
                input_tokens = 0
                tokens_out = 0
                tokens_total = 0
            elif self.framework in ['OpenAI', 'openAI', 
                                     'AzureOpenAI', 'Azure OpenAI', 'Azure_OpenAI', 'Azure-OpenAI', 
                                     'Ollama', 'ollama',
                                    ]: 
                # Get response contents from LLM agent output using OpenAI API. 
                response_content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                tokens_total = response.usage.total_tokens
            else:
                response_content = ''
                print(f"Warning: Information extraction from this response structure has not been implemented. Sorry!")
        return response_content    


    def get_tokens_consumed(self, response):
        """Get the tokens consumed by LLM from the response structure."""

        # Initialize 
        input_tokens = None
        output_tokens = None
        total_tokens = None
        
        if response:
            if not self.framework: 
                # how to get token usage generally? -- temporarily set to 0.
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
            elif self.framework in ['OpenAI', 'openAI', 
                                     'AzureOpenAI', 'Azure OpenAI', 'Azure_OpenAI', 'Azure-OpenAI', 
                                     'Ollama', 'ollama',
                                    ]: 
                # Get tokens consumed by LLM agent output using OpenAI API. 
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                print(f"Warning: Information extraction from this response structure has not been implemented. Sorry!")
        return input_tokens, output_tokens, total_tokens   

    
    def extract_answer(self, response):
        match = re.search(r'\n\n### Output:\s*((.*\n)+.*)$', response, re.DOTALL)
        if match:
            answer = match.group(1)
        else:
            # if no match, return empty string
            answer = ''        
        return answer


def run_llm():
    """Run LLM Agent"""

    agent = LLMAgent(framework='Ollama', model='llama3.3')
    sys_msg = 'You are an AI assistent helping with the given tasks. Help as much as possible.'
    prompt = 'Explain the structure of pathology reports.'
    response = agent.request(prompt=prompt, system_message=sys_msg, temperature=0)    
    answer = response_content = agent.get_response_content(response)
    print(answer)



## main ###################
if __name__ == "__main__":
    torch.cuda.empty_cache() 
    # call llm 
    fire.Fire(run_llm)
