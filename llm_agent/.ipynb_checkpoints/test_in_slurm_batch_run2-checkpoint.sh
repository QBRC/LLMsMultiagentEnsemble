#!/bin/bash
#SBATCH --job-name=llmp
#SBATCH --partition=GPUp4
#SBATCH --gres=gpu:1
#SBATCH --time=100-00:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user jingwei.huang@utsouthwestern.edu

# The standard output and errors from commands will be written to these files.
# %j in the filename will be replace with the job number when it is submitted.
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err


module load python/latest-3.11.x-anaconda
module load ollama

# mamba init
# mamba activate env_cuda121c
# conda activate /project/DPDS/Xiao_lab/shared/CondaEnv/ml_mmls2


# COMMAND GROUP 1
ollama serve &
# python llm_agent.py
curl --noproxy '*' http://localhost:11434/api/generate -d '{ "model": "llama3.1", "prompt": "Why is the sky blue", "stream": false }'