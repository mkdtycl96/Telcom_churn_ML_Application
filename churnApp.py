
import pickle
import pandas as pd
import streamlit as st

st.sidebar.title("Churn Probability of a Single Customer")
df=pickle.load(open("final_telco2","rb"))
df2 = pickle.load(open("df_show","rb"))
test_df = pickle.load(open("test","rb"))

model_xgb = pickle.load(open("xgb_model2","rb"))
model_log = pickle.load(open("log_model2_telco","rb"))
model_lgbm=pickle.load(open("lgbm_model_telco","rb"))
model_knn=pickle.load(open("knn_model_telco","rb"))
model_rf=pickle.load(open("rf_model","rb"))

html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">Churn Prediction </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

#st.write(df.head())
#st.table(df.head())
#st.dataframe(df.head())


models = st.sidebar.selectbox("Select Model",("Random Forest","XGBoost","Logistic","LGBM","KNN") )
tenure=st.sidebar.slider("Number of months the customer has stayed with the company (tenure)", 1, 72, step=1)
MonthlyCharges=st.sidebar.slider("The amount charged to the customer monthly", 0,100, step=5)
TotalCharges=st.sidebar.slider("The total amount charged to the customer", 0,5000, step=10)
Contract=st.sidebar.selectbox("The contract term of the customer", ( "Month-to-Month", 'One year',"Two year"))
PhoneService=st.sidebar.selectbox("Whether the customer has Internet service or not", ('No', 'Yes'))
InternetService=st.sidebar.selectbox("Customer’s internet service provider", ('DSL', 'Fiber optic', 'No'))

if models == "Random Forest":
    model = model_rf
elif models == "XGBoost" :
    model = model_xgb
elif models == "Logistic":
    model = model_log
elif models == "LGBM":
    model = model_lgbm
else:
    model= model_knn

def single_customer():
    my_dict = {"tenure" :tenure,
               "MonthlyCharges":MonthlyCharges,
               "TotalCharges": TotalCharges,
        "Contract": Contract,
        "PhoneService": PhoneService,
        "InternetService": InternetService
                 }
    columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_One year',
       'Contract_Two year', 'PhoneService_Yes', 'InternetService_Fiber optic',
       'InternetService_No']
    
    df_sample = pd.DataFrame.from_dict([my_dict])
    df_sample = pd.get_dummies(df_sample).reindex(columns=columns, fill_value=0)
    return df_sample

df1= single_customer()

if st.sidebar.button("Predict"):
    output = model.predict_proba(df1)[0]
    st.sidebar.success("The Churn probability of selected customer is % {}".format(round(output[0]*100,2)))

#st.table(df1)

st.success(model_rf.predict_proba(df1))
X= df.drop(["Churn"],axis=1)

#st.write(X)



if st.checkbox("Top Customers to Churn"):
    st.markdown("## How many cutomers to be selected?")
    aa = st.selectbox("Please select number of top customers to churn",(0,10,20,30,50,100,200))
    
    
    if aa:
        def top_loyal2():
            lst = []
            for i in model.predict_proba(test_df):
                lst.append(i[0])
            a = df2.loc[test_df.index]
            a["Churn"] = lst
            return a.sort_values(by="Churn",ascending=False).head(aa),lst 

        st.table(top_loyal2()[0])
        #st.write(top_loyal2()[1])
        
        
        
if st.checkbox("Top Loyal Customers"):
    st.markdown("## How many loyal cutomers to be selected?")
    aa = st.selectbox("Please select number of top loyal customers.",(0,10,20,30,50,100,200))
  
    if aa:
        def top_unloyal():
            lst = []
            for i in model.predict_proba(test_df):
                lst.append(i[0])
            a = df2.loc[test_df.index]
            a["Churn"] = lst
            return a.sort_values(by="Churn",ascending=True).head(aa),lst 

        st.table(top_unloyal()[0])
        #st.write(top_unloyal()[1])    
        
        
        
import random
if st.checkbox("Churn Probability of Randomly Customers"):
    st.markdown("## How many cutomers to be selected!?")
    aa = st.selectbox("Please select number of customers.",(0,10,15,20,25,30,40))

    if aa:
        def random_predict(a):
            rnd = random.choices(test_df.index, k = a)
            pred = model.predict_proba(test_df.loc[rnd])
            a = df2.loc[rnd]
            lst = []
            for i in pred:
                lst.append(i[0])
    
            a["Churn"] = lst
            return a
            
    
        st.table(random_predict(aa))
        
        
        
        
        





