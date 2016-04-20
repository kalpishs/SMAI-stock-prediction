____________________________________________________________________________________________________________________________________________
Mentor: Soham Saha

Project No. : 13

Team Name. : Pi-3.14159265 

By Kalpish Singhal, Megha Agarwal, Narendra Babu, Savalia Tejas
____________________________________________________________________________________________________________________________________________

==============================================================================================================================================
Stock Treand Forecasting   using   Supervised   Learning  methods
==============================================================================================================================================
Prediction of stock market is a long-time attractive topic to researchers from different fields. In particular,numerous studies have been conducted to predict the movement of stock market using machine learning algorithms such as support vector machine (SVM) and reinforcement learning. 
 In this project, we propose a new prediction algorithm that focus on Indian stock markets to predict the next-minute ,Next Day and Next week stock trend with the aid of SVM & Neural networks. Numerical results indicate a prediction accuracy of 80-85% For Hcltech,77-81% for Itc,69-74% for ONGC ,84-87% for Tcs,for 80-85% infy,81-86% for relliance using per minute prediction. 


==============================================================================================================================================
Data set Used
==============================================================================================================================================
BSE Sensex Dataset is used for all next-minute ,Next Day and Next week predictions using Yahoo finance Api .Online live data for the day is collected and stored for per minute data.We have a dataset from 6 year of daily Stock price of companies like TCS,Infosys,Ongc,Hcl tech,reliance industry.With the following Features.
->For miniute data :Timestamp,close,high,low,open,volume
->For Daily Data :Date,Open,High,Low,Close,Volume,Adj Close


==============================================================================================================================================
Project Scope
==============================================================================================================================================
This project is mainly focused on feature extraction  of Stock on the set of BSE. This will be divided into following  phases.

	(i) Feature Identification and extraction
	
	(ii) Model training using SVM & Neural Networks
	
	(iii) Testing using trained model.
	
	(iv) Evaluating the output obtained for different feature-models using metrics such as Precision, Recall and Accuracy.
	
	(v) Selecting some feature models and plot their Accuracy,Actual V/s predicted binary plot 

==============================================================================================================================================
List of Files 
==============================================================================================================================================

(i) hcltech.py,infy.py,itc.py,ongc.py,reliance.py (Python code to execute the minute Accurcy)

(ii) Stock_Prediction_Daily.py (main code for Day wise prediction)

(iii) metamap.py (for extracting semantic tags from Metamap(tool) generated output)

(iv) pos_tagger.py (for extracting part-of-speech tags from Tweet-NLP(tool) generated output)

(v) ortho.py (for assigning orthographic features)

(vi) cluster.py (for extracting cluster-id tags from Brown-Clustering(algorithm) generated output)

(vii) part.sh (for testing-exploring new feature sets from pre-existing feature file)

(viii) evaluate.py (for testing system generated output on the basis of different evaluation metrics)


==============================================================================================================================================
How to Run  
==============================================================================================================================================
->Install Following Tools:
	
1)Mallet :- A Java-based package for statistical natural language processing,text classification and  information extraction tool using Command line scripts.
  MALLET includes sophisticated tools for document classification: efficient routines for converting text to "features",a wide variety of algorithms eg. CRF
           Installation instructions available at http://mallet.cs.umass.edu/

2)Meta-Map 2013 :- MetaMap is a highly configurable program developed to map biomedical text to the UMLS Metathesaurus or,equivalently, to discover Metathesaurus concepts referred to in text. We have made use of Meta-Map Java-api for making use of this tool to find semantic tags.
	  Installation instructions available at https://metamap.nlm.nih.gov/JavaApi.shtml

3)Tweet-NLP :-  Provide a tokenizer, a part-of-speech tagger, hierarchical word clusters, and a dependency parser for tweet. We used it to address the problem of part-of-speech tagging for English data from the popular micro blogging service Twitter. The tool reports tagging results nearing 90% accuracy.
    Installation instructions available at http://www.cs.cmu.edu/~ark/TweetNLP/

4)You'll also need to run Brown-CLustering algorithm on both - testing and training data, for this project, we already have stored its output in usable format in the file IRE--Medical_NER_Twitter/data/cluster-50/paths Code and usage instructions available at https://github.com/percyliang/brown-cluster
 
Download the Project from here, https://github.com/kalpishs/IRE--Medical_NER_Twitter.git
Configure the file bash_5gram.sh in the codes_Phase2 folder in src, with the paths of installed tools, data, etc.
Start the metamap server, for Metamap server Java-api, run bash publc-mm/bin/mmserver13  (public-mm -> Metamap installed directory) 
Execute bash_5gram.sh and voila -> it will run successfully.
This bash script calls, 
All the files mentioned in the previous section sequentially, It also runs commands to get output from different tools such as:

1) bash $path/testapi.sh --input $files --output meta_out      (Running metamap)

2) bash $path2/runTagger.sh $files) > pos_out                  (Running tweet-nlp)

3) java -cp $java_home/class:$java_home/lib/mallet-deps.jar cc.mallet.fst.SimpleTagger --train true --model-file ../models/
   trained_model_5gram ../training_files/training_file_5gram   (Running mallet on training data and training model 'trained_model_5gram')

4) java -cp $java_home/class:$java_home/lib/mallet-deps.jar cc.mallet.fst.SimpleTagger --model-file ../models/trained_model_5gram 
   ../testing_files/testing_file_5gram) > ../system_result/system_tags_5gram  	(Running mallet on testing data using trained model)


==============================================================================================================================================
Output FORMAT
==============================================================================================================================================
We have used the BIO(Begin-Inside-Outside)  model for labelling entities. For a whole corpus of tweets, we are labelling data term-wise. This model (BIO) lets us handle entities that span over multiple terms.

For instance, the words “childhood asthma” represent a single entity which is a disease, In accordance with our  implementation, the separate words childhood and asthma will be labelled as

childhood -> Disease-Begin  and asthma -> Disease-Inside.

So, this is the output format and the corresponding labels are as follows:

{Disease-begin, Disease-inside, Drug-begin, Drug-inside, Symptom-begin, Symptom-inside, None}

So, we basically assign 2 labels (Begin, Inside) per entity (Disease, Drug, Symptom) and 1 extra label (None) representing Outside tag in BIO model.

These labels are the output which are obtained in system generated file by the trained tool.


==============================================================================================================================================
Conclusion
==============================================================================================================================================
The dataset provided to us consists of 85% of ‘None’ entity tags, hence if we label all the terms as none - we’ll have 85% accuracy. This conveys that accuracy is not the correct metric for correct evaluation of feature models.

Since this metric isn’t good-enough to tell us significance of one feature model over another, we tried using more application specific metrics such as Precision, Recall and F-Score. 

In this project we experimented with different feature sets and evaluated their efficiency in NER.

--- The best 3 feature models along with statistic analysis are represented in the presentation available at: <https://goo.gl/GN0AWr>

--- A video demonstrating our work has also been put-up and can be accessesed from: <https://youtu.be/dFKIy7CgMrg>

---Dropbox link :- <https://goo.gl/3Plc8s> 

---Github Web page <http://kalpishs.github.io/IRE--Medical_NER_Twitter>

------Slideshare: http://www.slideshare.net/NitishJain24/ire-presentation-team56


==============================================================================================================================================
Tags
==============================================================================================================================================
'Information Retrieval and Extraction Course', 'IIIT-H', Major Project', 'Mallet', 'Medical NER', 'Feature Sets', 'Disease', 'Drug', 'Symptom', 'Analysis and Approach'
