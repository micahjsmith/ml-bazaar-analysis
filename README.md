# ml-bazaar-analysis

Replication files for The Machine Learning Bazaar.

> M. Smith, C. Sala, J.M. Kanter, and K. Veeramachaneni. ["The Machine Learning Bazaar: Harnessing the ML Ecosystem for Effective System Development."](https://www.micahsmith.com/files/mlbazaar_sigmod20.pdf) SIGMOD 2020. 

## Usage

Run `make`, that's it.

```shell
make
```

This will check that you have [Docker](https://docs.docker.com/install/) 
installed and ask you to install it if you have not done so. Then, it will 
build a container that has Python 3.6 and all necessary dependencies. 
Finally, it will generate outputs from the Evaluation section and save them 
to the `./outputs` folder.

Alternately, you can install the requirements into a virtual environment and 
run the script directly, though reproducibility is less certain using this 
method due to variations in your system setup.

```shell
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python analysis.py
```
