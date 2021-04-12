# Implemented approaches
1) TCA original (see https://www.gipp.com/wp-content/papercite-data/pdf/hamborg2019a.pdf)
1) TCA with improved entity preprocessing 
2) XCoref 
2) XCoref + HC as a baseline to XCoref
3) EECDCR (see https://www.aclweb.org/anthology/P19-1409/)

# How to add a new entity identifier approach to newsalyze
1) create a new folder in ```newsalyze/entities```
2) create a constant with your approach's name in ```newsalyze/entities/const_dict_global.py```
3) create your entity identifier .py file where you will inherit ```Identifier``` class from ```newsalyze/entities/identifier.py```. 
Implement your entity identification approach there. The output needs to be a EntitySet object. To create entities, use 
default ```newsalyze/structures/entity.py``` and ```newsalyze/entities/entity_preprocessor.py```. You can also create subclasses 
from these classes, see examples in ```newsalyze/entities/msma```.
4) Create your class with configuration parameters that are required to execute your method. Inherit ```ParamsEntities``` 
from ```newsalyze/entities/params_entities.py```. Implement your save and read methods if you have a complicated configuration 
parameter structure. 
5) To add your method to newsalyze, add your class into ```newsalyze/pipeline/modules/entity_identifier.py``` into 
self.identifiers. Use the constant created at step 2. 
6) Add your approach into ```newsalyze/structures/configuration.py``` in self._run_config["entities"]. Use the same constant in the 
dictionary as in step 2. Also add your approach into self._run_config["candidates"]. You can either create your own setup 
for candidate extraction in ```newsalyze/candidates/params_cand.py``` or use ```ParamsCand(topic).get_default_params()```.
7) For the faster selection of your approach while debugging, on L49 and L50 in ```configuration.py```, add a constant of your method. 

We wish successful debugging!