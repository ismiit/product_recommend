import streamlit as st
import pandas as pd
import pickle
import re

similarity = pickle.load(open('similarity1.pkl','rb'))
def poster_string(ind):
    text = data2['image'][ind]
    text = str(text)
    text = text.replace('"','')
    pattern = re.compile('[\([{})\]]')
    text = pattern.sub(r'',text)
    pos = text.split(",")
    return pos[0]

def recommend(product):
    product_index = data2[data2['product_name'] == product].index[0]
    distances = similarity[product_index]
    product_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    posters_links = []
    indexes = []
    for x in product_list:
        recommended_movies.append(data2.iloc[x[0]].product_name)
        posters_links.append(poster_string(x[0]))
        indexes.append(x[0])

    return recommended_movies,posters_links,indexes

st.title('PRODUCT RECOMMENDATION SYSTEM')
data = pd.read_csv('data/flipkart_com-ecommerce_sample.csv')
data2 = data.iloc[0:5000,:]
product_names_list = data2['product_name'].values
selected_product = st.selectbox('Choose A product',product_names_list)

if st.button('Recommend'):
    products,posters,indx = recommend(selected_product)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(products[0])
        st.image(posters[0])
        st.write(data2['discounted_price'][indx[0]])
    with col2:
        st.text(products[1])
        st.image(posters[1])
        st.write(data2['discounted_price'][indx[1]])
    with col3:
        st.text(products[2])
        st.image(posters[2])
        st.write(data2['discounted_price'][indx[2]])
    with col4:
        st.text(products[3])
        st.image(posters[3])
        st.write(data2['discounted_price'][indx[3]])
    with col5:
        st.text(products[4])
        st.image(posters[4])
        st.write(data2['discounted_price'][indx[4]])