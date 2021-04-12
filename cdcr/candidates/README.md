# Candidate extraction settings
A candidate phrase or a candidate is a phrase which will be considered as a mention to an entity or a concept. A phrase 
may be a part of a frequent entity/concept or occur only once or twice and refer to an independent entity/concept.

Candidate extraction configuration include parameters that control: 
   * origin of candidates 
        * from annotated files
        * extracted from text
   * employment of coreference resolution (CR)
        * CR within a document
        * CR within a group of documents (usually 5 documents per group)
        * no CR
   * modification of a candidate 
        * a phrase extension with a parent NP from a parse tree, e.g., "a beautiful house" instead of "house"
        * change head of phrase in a phrase with quantifiers, e.g., "_hundreds_ of birds" to "hundreds of _birds_"
        
Both default candidate configuration and custom configuration preferred for entity identification methods exist 
in ```newsalyze/candidates/params_cand.py```. If you want to create your own configuration of candidate extractor, 
you can either create an initialising function in ```params_cand.py``` or save a default configuration into a json file,
 modify the file, and use configuration from the file. Details on declaration if config options can be found in 
 ```newsalyze/candidates/cand_enums.py```.
 
 To create a json file, the following steps are required: 
 1) start pipeline execution with ```start.py```
 2) choose a topic for execution
 3) choose "n" if you haven't executed newsalyze for this topic or "y" if you have some cache on the previsoua execution
    * if yes: select option 1 (or 0) to restore files not later than from preproprocessing step
 4) reply "n" to the questions about default parameters
 5) choose a entity identification method of your choice or a default one 
 6) choose a default candidate extraction method (option 0)
 7) reply "y" to the question if you want to save the config file 
 8) after you receive a message that your config was saved, you can interrupt the pipeline execution
 
 Now, you can modify the config file using parameter convention from ```newsalyze/candidates/cand_enums.py```. To use 
 the saved config, follow the steps 1-5 and choose your config at step 6. 
 
 **Important**: if you want to use the same 
 config for another topic, you will need to place the config json file into a similar folder structure in a topic folder 
 in ```resources/user_run_config```. 

## Requirements for annotated datasets
If you want to use for entity identification annotated candidates, you need to prepare a json file with the annotated 
mentions with a specified structure. Please, store all your conversion scripts in separate folder in ```./data```.

A json file needs to contain an array with dict objects. A dict needs to have the following attributes: 
```
"coref_chain": an entity name to which this mention refers, e.g., "Trump" ,
"tokens_str": a text of a mention, e.g., "Donald Trump",
"tokens_number": a list of token indexes within a sentence, e.g., [1, 2],
"sent_id": sentence index within a document, e.g., 0,
"doc_id": a file name of the orig document (e.g., 0_L),
"mention_id": a unique id of a mention, e.g., <doc_id>_<sent_id>_<headword_id>,
"mention_full_type": a type of mention, e.g., PERSON,
"mention_type": a shorter version of a mention type, e.g., PER,
```

Such a json file needs to be placed into ```./data/original/<topic_name>/annotation/<annotation_version>```, 
where ```topic_name``` is a name of the article collection for which the annotation is created, and ```annotation_version```
can be any name for the annotation version, e.g., original, or anything else if you had multiple annotation versions.

Make sure that you have ```self.origin_type = OriginType.ANNOTATED``` set in ```newsalyze/candidates/params_cand.py``` for 
your config generation method.