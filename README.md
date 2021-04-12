# XCoref: Cross-document coreference resolution (CDCR) in the wild

An end-to-end CDCR system aiming at resolving entity, event, and more abstract concepts with a word choice and labeling diversity
 from a set of related articles.  

### Environment and installation
Some troubleshooting information is found [here](INSTALLATION.md).
Clone the repository, install required packages and resources via pip. TODO

```
pip install requirements.txt
python setup.py
```

### Stanford CoreNLP Server
Next, execute the following to automatically set up the Stanford CoreNLP server and keep it running. It needs to be 
running during the main code execution. The start-up takes some time and only when finished, the newsalyze should be executed. 
Once the message `[main] INFO CoreNLP - StanfordCoreNLPServer listening at /0:0:0:0:0:0:0:0:9000` appears, you're ready to go.
```
python start_corenlp.py
```

### EECDCR: CDCR model by Barhom et al. 
If you want to use EECDCR as a method (see https://www.aclweb.org/anthology/P19-1409/) for entity identification module, follow the setup instructions [here](cdcr/entities/eecdcr/README.md).

## Run the analysis
To start the analysis:
```
python start_export_pipeline.py
```
After the pipeline was started, you will need to choose a collection of news articles, which you want to analyse. 

At the next question, choose "n" if you want to execute the pipeline from the very beginning and "y" if you have already 
executed the pipeline and it has cached the intermediate results, which you want to restore. 

Then, the pipeline will ask you to choose methods for the pipeline. To choose default parameters, answer "y". If you want to 
explore what are the other methods implemented for the pipeline modules, choose "n". For each module you will be offered a 
list of available methods. The default option will be marked in the selection list.

On a 64 GB of RAM and 2.8GHz, running the default pipeline on a small dataset of five news articles requires:
 1) TCA  ~25 minutes 
 2) XCoref ~50 minutes
 3) EECDCR ~ 6 hours
 
 ## Run the evaluation of CDCR methods
To replicate the numbers reported in the paper, start the following script:
```
python multiple_start.py
```
After the execution of the script is over, execute ```cdcr/util/evaluation/evaluation.py``` to collect the evaluation metrics. 
Warning! Requires a lot of RAM (>64 GB)! In case of out of memory error, comment L20 and run the script with, first,  uncommented L21 adn then uncommented L22 and commented L21.