# Target Concept Analysis (TCA) with improved preprocessing


Target Concept Analysis (TCA) is an approach for the entity identifier module in newsalyze. 
This folder contains code for the approach published in 

**F.Hamborg, A.Zhukova, B.Gipp "Automated Identification of Media Bias 
by Word Choice and Labeling in News Articles",  in Proceedings of the ACM/IEEE Joint Conference on Digital Libraries (JCDL), 2019**
(https://www.gipp.com/wp-content/papercite-data/pdf/hamborg2019a.pdf)

TCA takes extracted coreferential chains and noun phrases (NPs) and performs several merging iterations to categorize similar groups of phrases together 
(a phrase group can be of the size N>=1). The overall scheme of the merging approach resembles hierarchy, where the leaves on the lowest level are the extracted
coreferences and NPs, i.e., initially merged phrases, and each level in a hierarchy is a result of merging by each step. The output of the TCA is a 
list of K groups of semantically similar phrases, where K << N.

Before merging, we perform preprocessing and for each group of phrases we create an entity class that plays the role of a container for the 
phrases and extracted attributes (in the paper, we call entities as "WCL candidates"). For example, we extract all heads of contained phrases, determine an entity type, 
extract representative wordsets and phrases, etc. When one entity absorbs another entity, the attributes are recalculated to encounter the information
from the newly added phrases.

Each merging step follows a similar fashion of iterating over the entities: we sort the entities descendingly by their size (number of contained phrases), take the largest entity
and compare all other entities to it. If a smaller entity is similar to the considered entity, we merge phrases from the smaller entity to the bigger one, 
recalculate the attributes of the bigger entity, and remove the smaller entity from the list. We compare entities only with the entities of equal or smaller size, 
e.g., first with second, third, etc. ranked entities, then second with third, fourth, etc. 

The TCA consists of the following consecutive merging steps:
1) representative phrases' heads
2) sets of phrases' heads
3) representative labeling phrases
4) compounds
3) representative wordsets
4) representative frequent phrases


### 1: Merging using representative phrases' heads
_Representative phrase_ is an output flag-field of the coreference resolution that marks which phrase is considered to be semantically 
sufficient to represent the full chain, e.g., "President Trump": false, "Donald Trump": true, "the president": false. A representative phrase of an NP
is a phrase itself.

_Phrase's head_ is an attribute that each phrase in a coreference chain contains. For each NP we determine a phrase's head as the highest word in the dependency tree hierarchy.

We merge two entities if their heads of the representative phrases are similar by string comparison.

![Step 1](http://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/projects/Media_Bias_Analysis/Step1.png)


### 2: Merging using sets of phrases' heads
Unique heads of all phrases comprised in an entity form a _set of phrases' heads_. In the above-mentioned example, a set of phrases' heads are {Trump, president}. The method could merge 
such an entity to an entity where a set of headwords contain {billionaire}. 
To merge entities, we vectorize the sets into the Word2Vec word vector model, calculate mean word vector, and calculate cosine similarity on the mean vectors. 
If two entities are comparable, e.g., two entities belong to the same entity type, and the cosine similarity exceeds a certain threshold, we merge these entities. 
![Step 2](http://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/projects/Media_Bias_Analysis/Step2.png)


### 3: Merging using representative labeling phrases
The step aims at merging entities that contain phrases such as "illegal immigrants" and "undocumented immigrants."
The merging step includes three phases: 
1) determination of the representative labeling phrases 
2) construction of the similarity score matrix 
3) similarity estimation 

To determine _representative labeling phrases_, we extract all adjective+noun phrases by retrieving all "amod" relations from the dependency trees,
 then cluster vectorized labeling phrases with affinity propagation into the semantically similar groups of phrases, and pick as a representative of each group 
a labeling phrase with the highest global adjective frequency. 
 
We construct a score similarity matrix between representative labeling phrases of two entities by, first, calculating cosine similarity between two phrases 
and, second, converting a value into a score 0, 1 or 2, which represent similarity strength. 

Two entities are considered similar if the sum of all score matrix elements normalized by the matrix size exceeds a specific threshold. 
![Step 3](http://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/projects/Media_Bias_Analysis/Step3.png)


### 4: Merging using compounds

Merging using compound consists of two types:
1) common compounds similarity
2) compound-headword match

In the _common compound similarity_, we aim at merging entities containing phrases such as "DACA recipient" and "DACA applicant." First, we extract all phrases with "compound" relation in the dependency tree and determine if two entities have 
matching dependent named entity (NE) words. If they share common compounds, then we select compound phrases only with the common compounds and calculate
similarity score matrix similar to Step 3. Two entities are considered similar if the normalized sum of all matrix elements exceeds a specific threshold.
![Step 4-1](http://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/projects/Media_Bias_Analysis/Step4-1.png)

In the _compound-headword match_, we check if at least one NE-based dependent compound word of an entity is a head of at least one phrase in the second entity.
Using this method, we can merge "_Donald_ Trump" to "_Donald_."
![Step 4-2](http://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/projects/Media_Bias_Analysis/Step4-2.png)


### 5: Merging using representative wordsets
Each entity consists of multiple phrases, and if some phrases can be used several times in exactly the same words and some phrases can contain pattern noticeable in the wording. 

First, we exclude stopwords from each phrase and extract frequent itemsets from all phrases in an entity. For further calculation, we
use only maximal itemsets. Note that the word order in this method is disregarded.

To select _representative wordsets_, we calculate a representativeness score for each wordset that balances two factors: descriptiveness, i.e., a wordset would be as 
big as possible, and importance, i.e., the more often a wordset occurs in the phrases, the more significance it has. We select N=6 wordsets 
with the highest representativeness score. 

repr_score =  log(1 + itemset_length) * log(times_occurred)

Similar to Step 3, we calculate s score similarity matrix spanned over the representative wordsets of two entities. If the normalized sum 
of all matrix elements is higher than a specific threshold, we merge two entities.
![Step 5](http://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/projects/Media_Bias_Analysis/Step5.png)
 
 ### 6: Merging using the representative frequent phrases
 To identify some frequent patterns, especially in the long multi-word expressions, word order is required. For example, two entities with phrases 
 "Deferred Action of Childhood Arrivals" and "Childhood Arrivals" will not be merged at Step 5 but are subject for merging with the current step.
 
 First, we determine _representative frequent phrases_ by retrieving sequences of overlapping words among all phrases in an entity.
Similarly to the representative wordsets, we calculate the representativeness score and select N=6 most representative frequent phrases.
 
We calculate score similarity matrix spanned over representative frequent phrases and based on the Levenshtein distance.
In this case, we rate two phrases with a higher score if they have lower Levenshtein distance. 

In the final step, we calculate sums for each row and column normalized by their size, and if there is a value that exceeds a specific threshold, we merge 
two entities.
![Step 6](http://dke.uni-wuppertal.de/fileadmin/Abteilung/MT/projects/Media_Bias_Analysis/Step6.png)
