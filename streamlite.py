#Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lime import lime_tabular
import matplotlib.pyplot as plt


st.title("Customer Churn Prediction")

#Making sliders for the the input
st.sidebar.title("Input Customer Details")

gender = st.sidebar.selectbox(label = 'What is your gender?',
                             options = ('Male','Female'))

tenure = st.sidebar.number_input(label = 'Tenure (In Months)',
                         min_value = 0,
                         step = 1,
                         format = '%i')

SeniorCitizen = st.sidebar.selectbox(label = 'Are you a Senior Citizen?',
                             options = ('Yes','No'))

Partner = st.sidebar.selectbox(label = 'Do you have a partner?',
                       options = ('Yes','No'))

Dependents = st.sidebar.selectbox(label = 'Do you have any dependents to take care?',
                       options = ('Yes','No'))

PhoneService =  st.sidebar.selectbox(label = 'Do you use our phone service?',
                       options = ('Yes','No'))

MultipleLines = st.sidebar.selectbox(label = 'Do you use multiple lines?',
                             options = ('Yes', 'No', 'No phone service'))

InternetService = st.sidebar.selectbox(label = 'What kind of internet service do you use?',
                             options = ('Fiber Optic', 'DSL', 'No'))

onlineSecurity = st.sidebar.selectbox(label = 'Do you have online security for the internet?',
                             options = ('Yes', 'No', 'No internet service'))

onlineBackup = st.sidebar.selectbox(label = 'Do you have any online backup plan?',
                             options = ('Yes', 'No', 'No internet service'))

deviceProtection = st.sidebar.selectbox(label = "Do you have any device protection plan?",
                                options = ('Yes', 'No', 'No internet service'))

techSupport = st.sidebar.selectbox(label = "Have you accessed any Tech Support?",
                                options = ('Yes', 'No', 'No internet service'))

streamingTV = st.sidebar.selectbox(label = "Have you subscribed to Straming TV service?",
                                options = ('Yes', 'No', 'No internet service'))

streamingMovies = st.sidebar.selectbox(label = "Have you subscribed to Straming Movie service?",
                                options = ('Yes', 'No', 'No internet service'))

contract = st.sidebar.selectbox(label = "What kind of contract are you in?",
                                options = ('Month-to-Month', 'One year', 'Two year'))

paperlessBilling = st.sidebar.selectbox(label = "Have you opted for Paperless Billing?",
                                options = ('Yes','No'))

paymentMethod = st.sidebar.selectbox(label = "What kind of payment method do you use?",
                                options = ('Electronic Check','Mailed Check', "Bank Transfer (automatic)", "Credit Card (automatic)"))

MonthlyCharges = st.sidebar.number_input(label = 'What amount do you pay monthly?',
                         min_value = 0)

TotalCharge =  st.sidebar.number_input(label = 'What is your total charge?',
                         min_value = 0)
                                

#******************************************************************************
#******************************************************************************

#Convert the input data to the data-frame

input_df = pd.DataFrame({'tenure': [tenure], 'MonthlyCharges': [MonthlyCharges], 
                        'TotalCharge': [TotalCharge], 'gender': [gender], 
                        'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner],
                        'Dependents': [Dependents], 'PhoneService': [PhoneService],
                        'MultipleLines': [MultipleLines], 'InternetService': [InternetService], 
                        'onlineSecurity': [onlineSecurity],'onlineBackup': [onlineBackup], 
                        'deviceProtection': [deviceProtection], 'techSupport': [techSupport], 
                        'streamingTV': [streamingTV], 'streamingMovies': [streamingMovies],
                        'contract': [contract], 'paperlessBilling': [paperlessBilling], 
                        'paymentMethod': [paymentMethod]})

#making dummy data
input_df_dummy = pd.get_dummies(input_df, drop_first=True,
                                columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                                  'PhoneService', 'MultipleLines', 'InternetService', 'onlineSecurity',
                                  'onlineBackup', 'deviceProtection', 'techSupport', 'streamingTV',
                                  'streamingMovies', 'contract', 'paperlessBilling', 'paymentMethod'])

#******************************************************************************
#******************************************************************************
#Converting the input into repective data-frame of the model

if gender == 'Male':
    input_gender_Male = 1
else:
    input_gender_Male = 0

#**********************************************************
if SeniorCitizen == 'Yes':
    input_SeniorCitizen = 1
else:
    input_SeniorCitizen = 0
    

#***********************************************************

if Partner == 'Yes':
    input_Partner = 1
else:
    input_Partner = 0 
    
#**********************************************************
if Dependents == 'Yes':
    input_Dependent = 1
else:
    input_Dependent = 0

#***************************************************
if PhoneService == 'Yes':
    input_PhoneService = 1
else:
    input_PhoneService = 0

#*****************************************************
input_MultipleLines_No_service = 0
input_MultipleLines_Yes = 0

if MultipleLines == 'Yes':
    input_MultipleLines_Yes = 1
elif MultipleLines == "No phone service":
    input_MultipleLines_No_service = 1
    
#*********************************************************
input_internetServiceFiber_Yes = 0
input_internetService_No = 0

if InternetService == "Fiber Optic":
    input_internetServiceFiber_Yes = 1
elif InternetService == "No":
    input_internetService_No = 1

#**********************************************************
input_onlineSecurity_No_internet = 0
input_onlineSecurity_yes = 0

if onlineSecurity == 'No internet service':
    input_onlineSecurity_No_internet = 1
elif onlineSecurity == 'Yes':
    input_onlineSecurity_yes = 1

#************************************************************    
input_onlineBackup_No_internet = 0
input_onlineBackup_yes = 0

if onlineBackup == 'No internet service':
    input_onlineBackup_No_internet = 1
elif onlineBackup == 'Yes':
    input_onlineBackup_yes = 1

#*********************************************************************
    
input_DeviceProtection_No_internet = 0
input_DeviceProtection_yes = 0

if deviceProtection == 'No internet service':
    input_DeviceProtection_No_internet = 1
elif deviceProtection == 'Yes':
    input_DeviceProtection_yes = 1

#********************************************************************************
    
input_techSupport_no_internet = 0
input_techSupport_yes = 0

if techSupport == "No internet service":
    input_techSupport_no_internet = 1
elif techSupport == "Yes":
    input_techSupport_yes == 1

#*********************************************************************    
input_streamingTV_no_internet = 0
input_streamingTV_yes = 0

if streamingTV == 'No internet service':
    input_streamingTV_no_internet = 1
elif streamingTV == 'Yes':
    input_streamingTV_yes = 1

#*********************************************************************    
input_streamingMovie_no_internet = 0
input_streamingMovie_yes = 0

if streamingMovies == 'No internet service':
    input_streamingMovie_no_internet = 1
elif streamingMovies == 'Yes':
    input_streamingMovie_yes = 1

#*******************************************************************
    
input_contract_one_year = 0
input_contract_two_year = 0

if contract == 'One year':
    input_contract_one_year = 1
elif contract == 'Two year':
    input_contract_two_year = 1
    
#********************************************************************
if paperlessBilling == 'Yes':
    input_paperless_bill = 0
else:
    input_paperless_bill = 1
    
#***************************************************************

input_paymentMethod_credit = 0
input_paymentMethod_ElectronicCheck = 0
input_paymentMethod_mailedCheck = 0

if paymentMethod == 'Electronic Check':
    input_paymentMethod_ElectronicCheck = 1
elif paymentMethod == 'Mailed Check':
    input_paymentMethod_mailedCheck = 1
elif paymentMethod == "Credit Card (automatic)":
    input_paymentMethod_credit = 1
    
#***************************************************************************
#***************************************************************************
#**************************************************************************

input_column_name = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
       'SeniorCitizen_1', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No internet service', 'TechSupport_Yes',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

input_data_series = [tenure, MonthlyCharges, TotalCharge, input_gender_Male,
                     input_SeniorCitizen, input_Partner, input_Dependent, 
                     input_PhoneService, input_MultipleLines_No_service, 
                     input_MultipleLines_Yes, input_internetServiceFiber_Yes, 
                     input_internetService_No, input_onlineSecurity_No_internet, 
                     input_onlineSecurity_yes, input_onlineBackup_No_internet, 
                     input_onlineBackup_yes, input_DeviceProtection_No_internet, 
                     input_DeviceProtection_yes, input_techSupport_no_internet,
                     input_techSupport_yes, input_streamingTV_no_internet,
                     input_streamingTV_yes, input_streamingMovie_no_internet, 
                     input_streamingMovie_yes, input_contract_one_year, input_contract_two_year,
                     input_paperless_bill, input_paymentMethod_credit, input_paymentMethod_ElectronicCheck,
                     input_paymentMethod_mailedCheck]


#converting to numpy array
input_array = np.array(input_data_series).reshape(1,-1)

#converting to the pandas dataframe
df = pd.DataFrame([input_column_name, input_data_series])

#******************************************************************************
#UN-PICKLING THE MODEL
#******************************************************************************

pickle_in = open("rf.pkl", "rb")
classifier = pickle.load(pickle_in) 


# gbc_pickle = open("C:/Users/Sande/OneDrive/Desktop/Churn_Analysis/Churn_Analysis_CIS_630/gbc_model.pkl", "rb")



#******************************************************************************
#LIME Explainer
#******************************************************************************
#Reading the dataset for LIME
X_train_sm = pd.read_csv("train_data.csv")


#initiating the explainer instance
explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train_sm),
                                              feature_names = X_train_sm.columns, 
                                              class_names = ['Stay', 'Churn'],
                                              mode = 'classification')

#explaination for a specific example
exp = explainer.explain_instance(data_row=np.array(input_data_series),
                                 predict_fn = classifier.predict_proba)


#plotting the figure
plot = exp.as_pyplot_figure()

#******************************************************************************
#******************************************************************************
   
#Predicting the model results   
prediction = st.button('Predict Churn')

if prediction:
    
    prediction_class = classifier.predict(input_array)
    
    prediction_probability = classifier.predict_proba(input_array)
    
    #***************************************************************************
    #Plot to show the probability
    #********************************
    
    plot_data = pd.DataFrame([round(prediction_probability[0][0],3),
                              round(prediction_probability[0][1],3)],
                             ['No Churn', 'Churn'])
    
    c = ['red','green']
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1,])
    col = ["Not Churn", "Churn"]
    data = [round(prediction_probability[0][0],3)*100, 
            round(prediction_probability[0][1],3)*100]
    ax.bar(col,data, color=c)
    ax.set_ylim([0,100])
    plt.title("Probability of Churning",
              fontsize = 20)

    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    for i in range(len(data)):
        plt.annotate(str(data[i])+'%',  
                     xy=(col[i], data[i]), 
                     ha ="center", 
                     va="bottom",
                     fontsize = 16)
    
    #******************************************************************************
    #******************************************************************************
    if prediction_class[0] == 0:
        no_churn_prob = round(prediction_probability[0][0],3)*100
        st.write(f'The customer will not churn and the probability of not churning is {no_churn_prob}%')
        st.pyplot(fig)
        st.pyplot(plot)
        
    
    if prediction_class[0] == 1:
        churn_prob = round(prediction_probability[0][1],3)*100
        st.write(f'The customer will churn and the probability of retention is {churn_prob}%')
        st.pyplot(fig)
        st.pyplot(plot)
        

from PIL import Image
lime_exp = Image.open('lime_exp_info.png')
    

with st.expander("How to Read the Chart to understand the outputs?"):
     st.write("""Please use this plot to understand the LIME Explaination""")
     st.image(lime_exp)


















