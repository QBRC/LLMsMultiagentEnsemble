# LLMs Multiagent Ensemble (EnsemLLMs)

## Quick Start

### Prerequisites:

EnsemLLMs can run a group of large language models on OpenAI, AzureOpenAI, Ollama, and local deployments.

To run models on Ollama, you need to install Ollama first. See how to install Ollama at:

https://github.com/ollama/ollama

https://ollama.com/

### Create a EnsemLLMs application project

1. Create a project folder as "demo" shows.

2. Create prompt template file for the problem of interest in the project home folder. Create a system message file (optional) if prefer. 

&nbsp;  The variables of interest and their possible values, variable list, the expected JSON output template, can be embedded in a prompt template with the following place holders:

- `{{<Variable-Value Dictionary>}}` for variable-value table

- `{{<Variable List>}}` for variable list

- `{{<json_output_template>}}` for expected JSON output template

3. Create a variable-value dictionary in yaml, as demo show.

4. Specify expected flattern JSON output template in yaml, as demo show.

5. Specify configuration in "ensemble_config.yaml"

6. Modify setup.py for setting up the paths to local packages in EnsemLLMs.
   
### Examples

1. "./demo/demo.ipynb" provides an example to run EnsemLLMs for ECG report labeling, with a single case absert from MIMIC-IV ECG dataset.
  
2. "./example_project_small/demo_lung.ipynb" demos with a very small dataset about how to use EnsemLLMs to predict lung cancer TNM staging.

### License
Following UT Southwestern Office for Technology Development, the project is using the license from The University of Texas Southwestern Medical Center.
