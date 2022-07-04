### Problem
Churn analysis is all about decreasing the Churn Rate,  which is defined as the rate at which customers stop doing business with an entity. 
The same definition can be utilized in the context of employees for a corporation or company, students for a university, etc. 
The sole purpose of the analysis is to understand the customer through their buying behavior so that respective authorities can stop them from leaving said business/service. 
Nowadays, with the development of data-driven methodologies, various machine learning (ML) techniques automatically identify the customer with a high churn rate so that managerial personnel can take the necessary steps to decrease this rate. 
Artificial Intelligence (AI) automation increases efficiency and garners a quick response in decision-making with respect to churn rates. 

### Dataset
The dataset was extracted from the open-source dataset from a hypothetical telecommunication industry published by IBM that can be found via this [link](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
The datset consist of 7045 instances with 20 features. The target variable was 'Churn' which is binary consisting of 'Yes' if the customer churned/left or 'No' if the customer stayed. 
The dataset was also imbalanced consisting of 73.463 %  of 'No' and 26.537 % of 'Yes' values. SMOTE was used to oversample the minority class.

### Modeling & Result
Open Source Auto-ML- [PyCaret](https://pycaret.org/), was utilized to model. From the analysis, we found that the Random Forest outperformed any other model. Moreover, 
grid search on RF was performed to further improve the model performance. We were able to acheive accuracy of 76.61% and F1-score of 77%. 

### Explainability and Interpretability
For the model explanation, LIME (Local Interpretable Mode-Agnostic
Explanation) is used on the top of the random forest algorithm. LIME provides the
explanation for every single observation, explaining the features that do or do not support
the target variable Churn. 

### Deployment
The developed model and interpretability of the model is deployed as a web app through the use of [Streamlit](https://streamlit.io/). This app taked the input and outputs the
probability of the customer churn along side the LIME explainer showing the featurea supporting as well as contradicting the model result. The app can be accessed through this [link](https://share.streamlit.io/sharmajee499/churn_analysis_cis_630/main/streamlite.py)

Libraries Used: Numpy, Pandas, Scikit-Learn, PyCaret, LIME, Streamlit

### Libraries Installation
Install the required libraries by `pip install -r requirements.txt` in your anaconda console.
Make sure you have a different enviroment in conda for this project. 
