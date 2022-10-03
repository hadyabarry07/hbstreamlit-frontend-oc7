import streamlit as st
import requests
import pandas as pd
import io
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap 
import plotly.express as px
from sklearn.cluster import KMeans
import lightgbm
import ast 
plt.style.use('fivethirtyeight')


def main():
    

    # Set FastAPI backend
    backend = "https://oc-p7-fastapi.herokuapp.com/"

    df = pd.read_csv('application_train.csv',index_col='SK_ID_CURR')
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)

    sample = pd.read_csv('sample_X.csv',index_col='SK_ID_CURR')
    sample.drop(sample.filter(regex="Unname"), axis=1, inplace=True)

    description = pd.read_csv('columns_description.csv', usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

    id_client = sample.index.values

    @st.cache
    def load_model():
        clf = pickle.load(open('model.pkl', 'rb'))
        return clf

    #Loading selectbox
    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn

    @st.cache
    def load_infos_gen(df):
        lst_infos = [df.shape[0],
        round(df["AMT_INCOME_TOTAL"].mean(), 2),
        round(df["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = df.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets

    @st.cache
    def identite_client(df, id):
        data_client = df[df.index == int(id)]
        return data_client

    @st.cache
    def load_age_population(df):
        data_age = round((df["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache
    def load_kmeans(sample, id, mdl):
        index = sample[sample.index == int(id)].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, df], axis=1)
        return df_neighbors.iloc[:,1:].sample(10)

    @st.cache
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 


    #######################################
    # SIDEBAR
    #######################################

    #Title display
    html_temp = """<div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Tableau de bord modèle de Scoring </h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Décision d'octroi de crédit…</p>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

    #Customer ID selection
    st.sidebar.header("**General Info**")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    #Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(df) 

    ### Display of information in the sidebar ###
    #Number of loans in the sample
    st.sidebar.markdown("<u>Nombre de prêts dans l'échantillon :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    #Average income
    st.sidebar.markdown("<u>Revenu moyen en dollars (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    #AMT CREDIT
    st.sidebar.markdown("<u>Moyenne des prêts en dollars (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
        
    #PieChart
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)  


    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    #Display Customer ID from Sidebar
    st.write("Sélection de l'ID client :", chk_id)


    #Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Information client**")

    if st.checkbox("Affichage des informations client ?"):
            
        infos_client = identite_client(df, chk_id)
        st.write("**Sexe : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/365)))
        st.write("**Statut familial : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfants : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

        #Age distribution plot
        data_age = load_age_population(df)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)

            
        st.subheader("*Revenu (USD)*")
        st.write("**Revenu total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant du prêt : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Annuités du prêt : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        
        #Income distribution plot
        data_income = load_income_population(df)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)
        
        #Relationship Age / Income Total interactive plot 
        data_sk = df.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']/365).round(1)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                            size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                            hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                            title={'text':"Relationship Age / Income Total", 'x':0.5, 'xanchor': 'center'}, 
                            title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                            title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                            title="Income Total", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)

    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)


    # Upon upload of file (to test using test.csv from data/processed folder)


    X=sample.iloc[:, :-1]
    X = X[X.index == chk_id]

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    test_bytes_obj = io.BytesIO()
    X.to_csv(test_bytes_obj, index=True)  # write to BytesIO buffer
    test_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError

    files = {"file": ('sample_dataset.csv', test_bytes_obj, "multipart/form-data")}

    # Upon click of button
    if st.button('Start Prediction'):
        if len(X) == 0:
            st.write("Please upload a valid test dataset!") 
        else:
            with st.spinner('Prediction in Progress. Please Wait...'):
                output = requests.post(backend, 
                                        files=files,
                                        timeout=8000)
                
            st.success(output.json())
            # st.download_button(
            #     label='Download',
            #     data=json.dumps(output.json()), # Download as JSON file object
            #     file_name='prediction_results.json'
            #     )

    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(df, chk_id))

    
    #Feature importance / description
    if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
        shap.initjs()
        X = sample.iloc[:, :-1]
        X = X[X.index == chk_id]
        number = st.slider("Pick a number of features…", 0, 20, 5)

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)
        
        if st.checkbox("Need help about feature description ?") :
            list_features = description.index.to_list()
            feature = st.selectbox('Feature checklist…', list_features)
            st.table(description.loc[description.index == feature][:1])
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
            
    

    #Similar customer files display
    chk_voisins = st.checkbox("Afficher des clients similaires ?")

    if chk_voisins:
        knn = load_knn(sample)
        st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
        st.dataframe(load_kmeans(sample, chk_id, knn))
        st.markdown("<i>Target = Customer with default</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
        

    st.markdown('***')
if __name__ == '__main__':
    main()