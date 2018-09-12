1.KT_Project_Global0.java
	This is used for calculating the distance by using Global Edit Distance with basic parameter.
	* Parameter ‘namesFile’ in main class is absolute path of names.txt, which has to be set by users.
	* Parameter ‘trainFile’ in main class is absolute path of train.txt, which has to be set by users.
	* Parameter ‘resultFile’ in main class is absolute path of result.txt, which has to be set by users. If the result file is not exited, this system will create one.
	The method ‘match(String,String)’ is used to calculate the distance.
	The r parameter is [m, d, i, r]=[1,-1,-1,-1] 
	The format of result file is as follow:
	Persian name1: predicted name(s)
 	Persian name2: predicted name(s)
	…		…
	total_match:*** ; correct_match:*** ;precision:***
	match:*** ; correct_match:*** ;precision:***


2.KT_Project_Global1.java

	This is used for calculating the distance by using Global Edit Distance with modified parameter.
	* Parameter ‘namesFile’ in main class is absolute path of names.txt, which has to be set by users.
	* Parameter ‘trainFile’ in main class is absolute path of train.txt, which has to be set by users.
	* Parameter ‘resultFile’ in main class is absolute path of result.txt, which has to be set by users. If the result file is not exited, this system will create one.
	The method ‘match_pr(String,String)’ is used to calculate the distance.
	The r parameter is [m, d, i, r]=[1,-1,-1,(-1,0)]
	The format of result file is as follow:
	Persian name1: predicted name(s)
 	Persian name2: predicted name(s)
	…		…
	total_match:*** ; correct_match:*** ;precision:***
	match:*** ; correct_match:*** ;precision:***


3.KT_Project_Global2.java
	This is used for calculating the distance by using Global Edit Distance with another modified parameter.
	* Parameter ‘namesFile’ in main class is absolute path of names.txt, which has to be set by users.
	* Parameter ‘trainFile’ in main class is absolute path of train.txt, which has to be set by users.
	* Parameter ‘resultFile’ in main class is absolute path of result.txt, which has to be set by users. If the result file is not exited, this system will create one.
	The method ‘match_pr(String,String)’ is used to calculate the distance.
	The r parameter is [m, d, i, r]=[1,-1,-1,(-1,0,1)]
	The format of result file is as follow:
	Persian name1: predicted name(s)
 	Persian name2: predicted name(s)
	…		…
	total_match:*** ; correct_match:*** ;precision:***
	match:*** ; correct_match:*** ;precision:***


4.KT_Project_Local.java
	This is used for calculating the distance by using Local Edit Distance.
	* Parameter ‘namesFile’ in main class is absolute path of names.txt, which has to be set by users.
	* Parameter ‘trainFile’ in main class is absolute path of train.txt, which has to be set by users.
	* Parameter ‘resultFile’ in main class is absolute path of result.txt, which has to be set by users. If the result file is not exited, this system will create one.
	The method ‘match(String,String)’ is used to calculate the distance.
	The r parameter is [m, d, i, r]=[1,-1,-1,-1] 
	The format of result file is as follow:
	Persian name1: predicted name(s)
 	Persian name2: predicted name(s)
	…		…
	total_match:*** ; correct_match:*** ;precision:***
	match:*** ; correct_match:*** ;precision:***

5.KT_Projetc_NGram.java
	This is used for calculating the distance by using N-Gram algorithm.
	* Parameter ‘namesFile’ in main class is absolute path of names.txt, which has to be set by users.
	* Parameter ‘trainFile’ in main class is absolute path of train.txt, which has to be set by users.
	* Parameter ‘resultFile’ in main class is absolute path of result.txt, which has to be set by users. If the result file is not exiting, this system will create one.
	* Parameter ’n’ in main class is variable of N, the user can calculate different distance by changing this parameter
	The method ‘getDistance(String, String, int)’ is used to calculate the distance.
	The format of result file is as follow:
	Persian name1: predicted name(s)
 	Persian name2: predicted name(s)
	…		…
	total_match:*** ; correct_match:*** ;precision:***
	match:*** ; correct_match:*** ;precision:***

