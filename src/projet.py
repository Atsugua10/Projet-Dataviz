# Imports des bibliotheques:
import time
import datetime as dt
import streamlit as st
import numpy as np
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime as d
from PIL import Image
import altair as alt

# ==================================== # initialisation de l'application

@st.experimental_singleton(suppress_st_warning=True)
def start():
    st.title('*Author: TSAMPI MEZAGUE Augusta Merveille*')
    st.header('Projet: Augusta Immobilier - valeurs foncieres de 2016 a 2020')
    st.write('Cette application a pour but de vous accompagner dans la recherche d un bien imobilier, elle vous donne une idee leur prix moyen au metre carre par commune et par departement')
    st.write('Vous pouvez aussi specifier une region ou un type d appartement pour affiner vos recherches')

start()




#====================================== # Chargement des donnes
f5 = './data/df_sampled_2020.csv' 
f4 = './data/df_sampled_2020.csv' 
f3 = './data/df_sampled_2020.csv'
f2 = './data/df_sampled_2020.csv'  
f1 = './data/df_sampled_2020.csv' 

# f5 = './data/full_2020.csv' 
# f4 = './data/full_2019.csv' 
# f3 = './data/full_2018.csv' 
# f2 = './data/full_2017.csv' 
# f1 = './data/full_2016.csv' 

datas = [f5, f4, f3, f2, f1]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)

def init_df(datas):

     df_2016 = pd.read_csv(
          datas[0],
          header=0,low_memory=False)

     df_2017 = pd.read_csv(
          datas[1],
          header=0,low_memory=False)

     df_2018 = pd.read_csv(
          datas[2],
          header=0,low_memory=False)

     df_2019 = pd.read_csv(
          datas[3],
          header=0,low_memory=False)

     df_2020 = pd.read_csv(
          datas[4],
          header=0,low_memory=False)

     return df_2016, df_2017, df_2018, df_2019, df_2020

# Taux de valeurs manquantes:
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def missing_rate(df):
    miss_data = df.isna()
    miss_data = miss_data.sum()
    miss_data = miss_data/len(df)
    miss_data_bool = miss_data > 0.75

    return miss_data[miss_data_bool == True]
    
 # Suppression des données manquantes :
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def df_clean(df, indexes):
    df = df.drop(labels=indexes, axis=1)
    # j'affecte les moyennes aux elemnts vides
    df['nombre_pieces_principales'] = df['nombre_pieces_principales'].fillna(value=df['nombre_pieces_principales'].mean())
    df['surface_terrain'] = df['surface_terrain'].fillna(value=df['surface_terrain'].mean())
    df['valeur_fonciere'] = df['valeur_fonciere'].fillna(value=df['valeur_fonciere'].mean())
    
    
    for i in df.columns:
        if (df[i].isnull().values.any() == True):
            df.dropna(subset = [i,], inplace=True) # supression ligne si vide
    return df

# Transformation des types:
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def pre_process(df):
    for y in df.columns:
        if(df[y].dtype == object):
            df[y] = df[y].convert_dtypes()
    df['date_mutation'] = pd.to_datetime(df['date_mutation'])

    return df 

st.header("Chargement des donnees")


#================= Fonctions utiles ========================#
# Mois:
def get_mth(dt): 
    return dt.month

# Jour du mois:
def get_dom(dt): 
    return dt.day
 
    
# Nombre de lignes:
def get_rows(rows):
    return len(rows)

# Prix par mètre carré:
@st.cache(suppress_st_warning=True)
def get_price_by_square (df):
    return (df.valeur_fonciere / df.surface_terrain)


    
# Région:

# df_sampled_2020["code_departement"].unique() pour avoir les differentes valeurs

def get_regions (code_departement):
    auvergne = ['01','03','07','15','26','38','42','43','63','69','73','74']
    bourgogne = ['21','25','39','58','70','71','89','90']
    bretagne = ['22','29','35','56']
    val_loire = ['18','28','36','37','41','45']
    corse = ['20']
    ile_de_france = ['75','77','78','91','92','93','94','95']
    est = ['08','10','51','52','54','55','57','67','68','88']
    haut_france = ['02','59','60','62','80']
    normandie = ['14','27','50','61','76']
    aquitaine = ['16','17','19','23','24','33','40','47','64','79','86','87']
    occitanie = ['09','11','12','30','31','32','34','46','48','65','66','81','82']
    pays_loire = ['44','49','53','72','85']
    provence = ['04','05','06','13','83','84']
    

    region =''

    if (code_departement in auvergne):
        region = "AUVERGNE-RHÔNE-ALPES"

    elif (code_departement in bourgogne):
        region = "BOURGOGNE-FRANCHE-COMTÉ"
    
    elif (code_departement in bretagne):
        region = "BRETAGNE"
    
    elif (code_departement in val_loire):
        region = "CENTRE-VAL-DE-LOIRE"
    
    elif (code_departement in corse):
        region = "CORSE"

    elif (code_departement in ile_de_france):
        region = "ÎLE-DE-FRANCE"

    elif (code_departement in est):
        region = "GRAND-EST"

    elif (code_departement in haut_france):
        region = "HAUT-DE-FRANCE"

    elif (code_departement in normandie):
        region = "NORMANDIE"

    elif (code_departement in aquitaine):
        region = "NOUVELLE-AQUITAINE"

    elif (code_departement in occitanie):
        region = "OCCITANIE"

    elif (code_departement in pays_loire):
        region = "PAYS-DE-LA-LOIRE"

    elif (code_departement in provence):
        region = "PROVENCE-ALPES-CÔTE D'AZUR"
    
    else:
        region = "CORSE"

    return region

def process_square(df):

    # On regroupe les données selon le prix par mètre carré moyen:
    group_by_square = df.groupby(['code_departement','region']).agg({'surface_terrain':'mean','valeur_fonciere':'mean'})
    
    return group_by_square


#================================ tris
#Selon la commune
@st.cache(suppress_st_warning=True)
def per_commune(df):
    
    # On regroupe les données par commune:
    group_by_commune = df.groupby('nom_commune').size()
    group_by_commune = group_by_commune.sort_values()

    return group_by_commune

#pricing par commune 
@st.cache(suppress_st_warning=True)
def per_commune_square(df):

    # On regroupe les données selon le prix par mètre carré moyen par commune:
    group_by_commune_square = df.groupby('nom_commune').agg({'price_by_square':'mean'})
    group_by_commune_square = group_by_commune_square.sort_values(by='price_by_square')
    
    return group_by_commune_square

#pricing par commune en fonction du departement choisi
@st.cache(suppress_st_warning=True)
def per_commune_departement_square(df):

    # On regroupe les données selon le prix par mètre carré moyen:
    group_by_commune_departement_square = df.groupby(['code_departement','nom_commune']).agg({'price_by_square':'mean'})

    return group_by_commune_departement_square

#selon le departement
@st.cache(suppress_st_warning=True)
def per_departement(df):

    # On regroupe les données par département:
    group_by_departement = df.groupby('code_departement').size()
    group_by_departement = group_by_departement.sort_values()
    
    return group_by_departement

#pricing par departement
@st.cache(suppress_st_warning=True)
def per_departement_square(df):

    # On regroupe les données selon le prix par mètre carré moyen:
    group_by_departement_square = df.groupby('code_departement').agg({'price_by_square':'mean'})
    group_by_departement_square = group_by_departement_square.sort_values(by='price_by_square')
    
    return group_by_departement_square




# ================= selon la periode
@st.cache(suppress_st_warning=True)
def per_month(df):
    group_by_dom = df.groupby('dom').size()
    
    return group_by_dom

#selon la region
@st.cache(suppress_st_warning=True)
def per_region(df):
    group_by_departement_region = df.groupby(['region','code_departement']).size().unstack(level=0)
    group_by_departement_region = group_by_departement_region.fillna(0)
    return group_by_departement_region

#type de local en fonction du departement
@st.cache(suppress_st_warning=True)
def per_local(df):
    group_by_local = df.groupby(['code_departement','type_local']).size().unstack(level=1)
    group_by_local = group_by_local.fillna(0)
    return group_by_local

#type de local en fonction de la region
@st.cache(suppress_st_warning=True)
def per_local_region():
    group_by_local_region = df.groupby(['region','type_local']).size().unstack(level=1)
    group_by_local_region = group_by_local_region.fillna(0)
    return group_by_local_region

#traitement des donnees
@st.cache(suppress_st_warning=True)
def processing(df):
    
    df['mth']  = df['date_mutation'].map(get_mth)
    df['dom']  = df['date_mutation'].map(get_dom)
    df['price_by_square'] = get_price_by_square(df)
    df['region']  = df['code_departement'].map(get_regions)
    
    return df


# ===================================== # naviguation du client
image = Image.open('image0.jpg')
st.sidebar.image(image, caption=None)
st.sidebar.header("Navigation")
option_data = st.sidebar.selectbox( "Selectionner l'annee de votre choix ", ['Valeurs Foncières 2016','Valeurs Foncières 2017','Valeurs Foncières 2018', 'Valeurs Foncières 2019', 'Valeurs Foncières 2020'])


def init_data():
    df_2016, df_2017, df_2018, df_2019, df_2020 = init_df(datas)

    if (option_data == 'Valeurs Foncières 2016'):
        st.markdown('Data set de 2016')
        df = df_2016
        if st.button('See the 2016 data'):
            st.write('Dataframe réduit:')
            st.dataframe( df.head(10))
            st.text("La taille du dataframe séléctionné est :" + str(df.shape))

    elif (option_data == 'Valeurs Foncières 2017'):
        st.markdown('Data set de 2017')
        df = df_2017
        if st.button('See the 2017 data'):
            st.write('Dataframe réduit:')
            st.dataframe( df.head(10))
            st.text("La taille du dataframe séléctionné est :" + str(df.shape))

    elif (option_data == 'Valeurs Foncières 2018'):
        st.markdown('Data set de 2018')
        df = df_2018
        if st.button('See the 2018 data'):
            st.write('Dataframe réduit:')
            st.dataframe( df.head(10))
            st.text("La taille du dataframe séléctionné est :" + str(df.shape))

    elif (option_data == 'Valeurs Foncières 2019'):
        st.markdown('Data set de 2019')
        df = df_2019
        if st.button('See the 2019 data'):
            st.write('Dataframe réduit:')
            st.dataframe( df.head(10))
            st.text("La taille du dataframe séléctionné est :" + str(df.shape))

    elif (option_data == 'Valeurs Foncières 2020'):
        st.markdown('Donnee de 2020')
        df = df_2020
        if st.button('See the 2020 data'):
            st.write("Data frame reduit: ", df.head(10))
            st.text("La taille du dataframe séléctionné est :" + str(df.shape))

    
    col1, col2, col3 = st.columns(3)

    with col1:
            pass
    with col3:
            pass
    with col2 :
            btn = st.button('Voir le data frame traité')

    if btn:

            miss = missing_rate(df)
            df = df_clean(df, miss.index)
            df = pre_process(df)
            df = processing(df)

            agree = st.checkbox('Voir moins')

            if agree:
                    btn = False
           
            st.caption('Dataframe final')
            st.dataframe(df.head())
    
    
    return df, df_2016, df_2017, df_2018, df_2019, df_2020           

df, df_2016, df_2017, df_2018, df_2019, df_2020 = init_data()

def init_sidebar(df):

        st.header('Visualisation des prix des biens fonciers (en metre carre)')

        df['region']  = df['code_departement'].map(get_regions)
        df['price_by_square'] = get_price_by_square(df)

        options = st.sidebar.multiselect('Quelle region vous interesse?',
        ["AUVERGNE-RHÔNE-ALPES", "BOURGOGNE-FRANCHE-COMTÉ", "BRETAGNE", "CENTRE-VAL-DE-LOIRE", "CORSE",
        "ÎLE-DE-FRANCE", "GRAND-EST","HAUT-DE-FRANCE","NORMANDIE", "NOUVELLE-AQUITAINE", "OCCITANIE", "PAYS-DE-LA-LOIRE", "PROVENCE-ALPES-CÔTE D'AZUR"])
        
        if options:
                mask1 = df['region'].isin(options)
                df = df[mask1]
                
        else:
                options = 1
        return options

options = init_sidebar(df)


def init_square(df):

        df['region']  = df['code_departement'].map(get_regions)
        df['price_by_square'] = get_price_by_square(df)

        if (type(options) != int):
            mask1 = df['region'].isin(options)
            df = df[mask1]

        group_by_commune_square = per_commune_square(df)
        group_by_commune_square.name = 'Prix moyen'
  
        group_by_departement_square = per_departement_square(df)
        group_by_departement_square.name = 'Prix moyen'

        col1, col2 = st.columns(2)
        col1.metric("Ville avec le prix moyen le plus haut:",group_by_commune_square.tail(1).last_valid_index(), str(group_by_commune_square.values[-1:]).strip('[]') + "€")
        col2.metric("Département avec le prix moyen le plus haut:", group_by_departement_square.tail(1).last_valid_index(), str(group_by_departement_square.values[-1:]).strip('[]') + "€")

        st.caption("Prix moyen par commune - Top 10")
        st.bar_chart(group_by_commune_square.tail(10))

    
        st.caption("Prix moyen par departement - Top 10")
        st.bar_chart(group_by_departement_square.tail(10))
    
        group_by_square = process_square(df)
        group_by_square  = group_by_square.reset_index()

init_square(df)



def init_charts(df):

    st.subheader('Visualisation des données sélectionnées')
    group_by_commune = per_commune(df)
    group_by_commune.name = 'Transactions'
    group_by_departement = per_departement(df)
    group_by_departement.name = 'Transactions'
    group_by_commune_square = per_commune_square(df)
    group_by_commune_square.name = 'Prix moyen'
    

    col1, col2 = st.columns(2)
    col1.metric("Commune avec le plus de transactions  :",group_by_commune.idxmax(), group_by_commune.tail(1).item())
    col2.metric("Département avec le plus de transactions :", group_by_departement.idxmax(), group_by_departement.tail(1).item())
    
    
    st.caption(" Top 10 des communes les moins cheres")
    st.area_chart(group_by_commune_square.tail(10))


    fig1 = plt.figure()
    x1 = group_by_commune.tail(10)
    ax = x1.plot(kind="bar", label = "Top 10 communes", width=0.5)
    ax.set_xlabel('Communes')
    ax.set_ylabel('Fréquence transaction')
    ax.set_title('Fréquence par commune - Top 10')


    fig2 = plt.figure()
    x2 = group_by_departement.tail(10)
    ax = x2.plot(kind="bar", label = "Top 10 départements", width=0.5)
    ax.set_xlabel('Département')
    ax.set_ylabel('Fréquence transaction')
    ax.set_title('Fréquence par département - Top 10')

    col1, col2 = st.columns([3,2])
    col1.caption('Histogramme trié des fréquences par commune')
    col2.caption('Dataframe des fréquences par commune')

    col1, col2 = st.columns([3,2])
    col1.pyplot(fig1)
    col2.write(group_by_commune.tail(10))
    
    col1, col2 = st.columns([2,3])
    col2.caption('Histogramme trié des fréquences par département')
    col1.caption('Dataframe des fréquences par département')

    col1, col2 = st.columns([2,3])
    col2.pyplot(fig2)
    col1.write(group_by_departement.tail(10))

    
    st.header('Transaction selon le type de local')

    group_by_local = per_local(df)
    group_by_local.columns = group_by_local.columns.values
    st.caption('Fréquences des transactions par région selon le type de local')
    st.line_chart(group_by_local)

    # loc = df['type_local'].unique()

    # option_loc = st.selectbox('Selectionnez un type de local', loc)
    # df = df[df['type_local'] == option_loc].sort_values(by='region')

    # st.subheader('repartition suivant les regions du type de local ')

    # df = df[['type_local','region']].groupby(['type_local','region']).count().reset_index()
    # df_type_local = df[df['region']== option_loc]

    # fig1 = px.pie(df, names = 'type_local', values='region', color='region')
    # st.plotly_chart(fig1, use_container_width=True)

    

init_charts(df)

def init_map(df):

    st.header('Visualisation des differents bien fonciers sur la map')
    
    fig3 = px.scatter_mapbox(df, lat= "latitude", lon="longitude",hover_data=['longitude','latitude','code_departement'],

            color_discrete_sequence=["purple"], zoom=5).update_traces(marker=dict(size=5))

    fig3.update_layout(mapbox_style="open-street-map") 

    st.plotly_chart(fig3)

init_map(df)
