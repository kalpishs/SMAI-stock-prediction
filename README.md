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

(ii) Stock_Prediction_Daily_noofdays_allcompanies.py (main code for Day wise prediction)

(iii) The Plots folder contains plotted comparisions between predicted and actual trends for minute wise prediction, along with the accuracy change as more data rows are added to the existing dataset in the Real_Time_Accuracy folder 

(iv) daily_dataset folder contains the dataset used for inter-day prediction whereas the dataset folder contains the dataset used for intra-day prediction.



==============================================================================================================================================
How to Run  
==============================================================================================================================================
->Install Following Tools:
	
1)Numpy
2)Matplotlib
3)Sci-kit learn
4)PyBrain

To see the results on intra-day training, run individual files, named by companies.
To see the results on inter-day prediction, run the file Stock_Prediction_Daily_noofdays_allcompanies.py


==============================================================================================================================================
Output FORMAT
==============================================================================================================================================
For all files, we get accuracies for 4 sets of features which can be then analysed and the best set can be suggested for prediction in general. 
we plot two step graphs, one for predicted binary values depicting uptrends and downtrends and the same for the actual values for providing a visual comparision aid.



==============================================================================================================================================
Tags
==============================================================================================================================================
'Statistical Methods in Artificial Intelligence Course', 'IIIT-H', Major Project', 'Stock Prediction', 'SVM', 'Neural Networks', 'Sci-Kit Learn', 'Pybrain', 'Classification on Indian Stocks'
