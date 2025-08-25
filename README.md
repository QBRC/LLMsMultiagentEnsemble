# LLMs Multiagent Ensemble (EnsemLLMs)

## Quick Start

### Prerequisites:

EnsemLLMs can run a group of large language models on OpenAI, AzureOpenAI, Ollama, and local deployments.

To run models on Ollama, you need to install Ollama first. See how to install Ollama at:

https://github.com/ollama/ollama

https://ollama.com/

### Create a EnsemLLMs application project

Create a project folder as "demo" shows.

Create prompt template file for the problem of interest in the project home folder. Create a system message file (optional) if prefer. 

Create a variable-value dictionary in yaml, as demo show.

Specify expected flattern JSON output template in yaml, as demo show.

Specify configuration in "ensemble_config.yaml"

Modify setup.py for setting up the paths to local packages in EnsemLLMs.

"./demo/demo.ipynb" provides an example to run EnsemLLMs for ECG report labeling.



