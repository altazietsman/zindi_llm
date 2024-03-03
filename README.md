# zindi_llm

This repository demonstrated how to create an  AI assistant capable of providing knowledge contained in the Malawi Technical Guidelines for Integrated Disease Surveillance and Response (TGs for IDSR).

Models were required to run on cpu. Qauntized models were therefor used.

# ENV

Python 3.11.7 was used.  All requirements can be found in the requirements.txt file. Ensure that all libraries are installed (locally or within a conda env.)

To install from requirements file run:
pip3 install -r requirements.txt

If you are struggling to see your conda env in your notebook, try also installing nb_conda_kernels (conda install nb_conda_kernels) as well.

It is important set your PYTHONPATH to this dir for example:

export PYTHONPATH=${PYTHONPATH}:<'path_to_directory'>

## Ollama model ENV

To use ollama install follow these instructions:

- Download and install Ollama onto the available supported platforms (including Windows Subsystem for Linux) (https://ollama.com/)
- Fetch available LLM model via ollama pull llama2 or ollama pull phi (or any model of choice.)

This will download the default tagged version of the model. Typically, the default points to the latest, smallest sized-parameter model.

## llama model ENV

To use llama follow these instructions:

NB: please note that depending on whether a gglm model or gguf model is used different version of llaam-cpp-python should be used. 

Examples given to pull llama2 models however any gguf or gglm model version can be utilised.

### gguf model versions 
- Download: huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q5_K_M.gguf --local-dir . --local-dir-use-symlinks False

- Then run: pip install llama-cpp-python==0.2.39 

### gglm model versions

- Download: `wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_1.bin`
- Then run: pip install llama-cpp-python==0.1.78

# Example notebooks
To run the ollama models see the ollama_example.pynb notebook.

# Data
The  Technical Guidelines can be found under data. In total 6 books were used to retrieve information.

# Submission File
The submission file used can be found in the submissions dir

# Models
All model classes tested can be found in the models directory. Note that not all models were used for the final submission, however they are still supplied for transparency. 

# Repo structure
- zindi lmm

    - data
        - booklets
        - submissions
        - Train.csv
        - Test.csv
    - models
        - llama.py
        - ollama.py
   
    - utils
        - embeddings.py
        - postprocessing.py
        - preprocess.py
        - response_generator.py
        - utils.py
        - vector_store.py
    - ollama_example.ipynb
    - README.md
    - requirements.text
