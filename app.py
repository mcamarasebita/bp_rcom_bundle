from itertools import accumulate
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

st.set_page_config(
    page_title="Bundle recom", page_icon="📊", initial_sidebar_state="expanded"
)

st.title("Assessment_Exploratory_analysis Dashboard")
# Tops 10 User, subscib and revenu
df_BTL_msisdn_id=pd.read_csv('./df_data_BTL_msisdn.csv', sep=',',encoding= 'utf-8')
df_msisdn_id=pd.read_csv('./top_bundle_msisdn.csv', sep=',',encoding= 'utf-8')
df_subscriptions=pd.read_csv('./top_bundle_subscriptions.csv', sep=',',encoding= 'utf-8')
df_BTL_subscriptions=pd.read_csv('./df_data_BTL_subscriptions.csv', sep=',',encoding= 'utf-8')
df_BTL_sum_revenu=pd.read_csv('./df_data_BTL_sum_revenu.csv', sep=',',encoding= 'utf-8')
df_sum_revenu=pd.read_csv('./top_bundle_sum_revenu.csv', sep=',',encoding= 'utf-8')
#Dataset of unsubs
df_recom_2bundle_unsubs=pd.read_csv('./unsubs_recomend_databtl.csv', sep=',',encoding= 'utf-8')
# Data for col_filter
cf_final_dataset=pd.read_csv('./cf_final_dataset.csv', sep=',',encoding= 'utf-8')
# Data agreg BTL : price of BTL and user_Id, bundle_Id
price=pd.read_csv('./bundle_BTL.csv', sep=',',encoding= 'utf-8')
df_kmeans= pd.read_csv('./df_kmeans.csv',sep = ',')


# preparation for viewing
df_msisdn_id = df_msisdn_id.reset_index()
fig_df_msisdn_id = px.bar(df_msisdn_id, x='bundle_name', y='msisdn_id', title="df_msisdn_id Global")
df_BTL_msisdn_id = df_BTL_msisdn_id.reset_index()
fig_df_BTL_msisdn_id = px.bar(df_BTL_msisdn_id, x='bundle_name', y='msisdn_id', title="df_BTL_msisdn_id Global")
df_BTL_subscriptions = df_BTL_subscriptions.reset_index()
fig_df_BTL_subscriptions = px.bar(df_BTL_subscriptions, x='bundle_name', y='sum_subscriptions', title="df_BTL_subscriptions Global")
df_subscriptions = df_subscriptions.reset_index()
fig_df_subscriptions = px.bar(df_subscriptions, x='bundle_name', y='sum_subscriptions', title="df_subscriptions Global")
df_BTL_sum_revenu = df_BTL_sum_revenu.reset_index()
fig_df_BTL_sum_revenu = px.bar(df_BTL_sum_revenu, x='bundle_name', y='sum_revenu', title="df_BTL_sum_revenu Global")
df_sum_revenu = df_sum_revenu.reset_index()
fig_df_sum_revenu = px.bar(df_sum_revenu, x='bundle_name', y='sum_revenu', title="df_sum_revenu Global")

st.markdown("##### Top 10 Bundlle by user Globaly")
with st.expander("Top 10 Bundlle by user Globaly"):
    # columnar division 
    col1, col2 = st.columns(2)

    col1.markdown("###### Top 10 by users Bundlle Globaly")
    col2.markdown("###### Top 10 by users Bundlle BTL Globaly")

    col1.dataframe(df_msisdn_id)
    col2.dataframe(df_BTL_msisdn_id)

    col1.plotly_chart(fig_df_msisdn_id)
    col2.plotly_chart(fig_df_BTL_msisdn_id)

st.markdown("##### Top 10 Bundlle by subscrib Globaly")
with st.expander(" Top 10 Bundlle by subscib Globaly"):
    col1, col2 = st.columns(2)

    col1.markdown("###### Top 10 by subscib Bundlle Globaly")
    col2.markdown("###### Top 10 by subscib Bundlle BTL Globaly")

    col1.dataframe(df_subscriptions)
    col2.dataframe(df_BTL_subscriptions)

    col1.plotly_chart(fig_df_BTL_subscriptions)
    col2.plotly_chart(fig_df_subscriptions)

st.markdown("##### Top 10 Bundlle by revenu Globaly")
with st.expander(" Top 10 Bundlle by revenu Globaly"):
    col1, col2 = st.columns(2)

    col1.markdown("###### Top 10 by reveny Bundlle Globaly")
    col2.markdown("###### Top 10 by reveny Bundlle BTL Globaly")

    col1.dataframe(df_sum_revenu)
    col2.dataframe(df_BTL_sum_revenu)

    col1.plotly_chart(fig_df_sum_revenu)
    col2.plotly_chart(fig_df_BTL_sum_revenu)

#######################################################################################################################################
##                                                                                                                                   ##
###                                             Bundle_collaboratif_filtering_engine                                                ###
##                                                                                                                                   ##
#######################################################################################################################################

st.markdown("### Recommendation for unsubscribe customers ")
# tto select a user for non-subscribers
aa= st.selectbox("Select USer", pd.unique(df_recom_2bundle_unsubs["msisdn_id"]))
#Propose 2 bunble 
if aa:
    df_recom_2bundle_unsubs1 = df_recom_2bundle_unsubs[df_recom_2bundle_unsubs['msisdn_id'] == aa]
    st.dataframe(df_recom_2bundle_unsubs1[['msisdn_id','Data_BTL_offers_1','Data_BTL_offers_2']])


st.markdown("### Bundle_collaboratif_filtering_engine ")
with st.container():
    final_dataset= cf_final_dataset[['bundle_name','id_bundle','msisdn_id_x','sum_subscriptions']]
    df_user_item_matrix = final_dataset.pivot(index='id_bundle',columns='msisdn_id_x',values='sum_subscriptions').fillna(0)

    #Creation Matrice
    sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
    sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
    csr_sample = csr_matrix(sample)
    csr_data = csr_matrix(df_user_item_matrix.values)
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)
    df_user_item_matrix.reset_index(inplace=True)

    # Function to recommend content with K-NN
    def get_bundle_recommendation(movie_name):
                n_movies_to_reccomend = 10
                movie_list = final_dataset[final_dataset['bundle_name'].str.contains(movie_name)]  
                if len(movie_list):        
                    movie_idx= movie_list.iloc[0]['id_bundle']
                    movie_idx = df_user_item_matrix[df_user_item_matrix['id_bundle'] == movie_idx].index[0]

                    distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
                    rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),\
                                               key=lambda x: x[1])[:0:-1]
        
                    recommend_frame = []

                    for val in rec_movie_indices:
                            movie_idx = df_user_item_matrix.iloc[val[0]]['id_bundle']
                            idx = final_dataset[final_dataset['id_bundle'] == movie_idx].index
                            recommend_frame.append({'bundle_name':final_dataset.iloc[idx]['bundle_name'].values[0],'Distance':val[1]})
                    dfx = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
                    return dfx
    
                else:
                    return "No movies found. Please check your input"
# Select user
aaa= st.selectbox("Select USer", pd.unique(cf_final_dataset["msisdn_id_x"]))

if aaa:
    df_recom = cf_final_dataset[cf_final_dataset['msisdn_id_x'] == aaa]
    # Get the most subscribed bundle
    df_recom1=df_recom[df_recom['sum_subscriptions']==df_recom['sum_subscriptions'].max()]
    a=df_recom1['bundle_name'].iloc[0]
    #Recover its price
    price_a=df_recom1['price_point'].iloc[0]

    user_Nbr_subs=df_recom.groupby(['msisdn_id_x'])['sum_subscriptions'].sum()
    user_Nbr_subs=pd.DataFrame(user_Nbr_subs)
    user_Nbr_subs=user_Nbr_subs['sum_subscriptions'].iloc[0]
    users_bunble_maxSubs=df_recom.sort_values(by=['sum_subscriptions'],ascending=False)
    users_bunble_maxSubs=users_bunble_maxSubs[['bundle_name','sum_subscriptions']]


    user_Nbr_subs=int(user_Nbr_subs)
    col1, col2 = st.columns(2)

    col1.metric(label="Total subscriptions", value=user_Nbr_subs,
        delta_color="inverse")
    col2.write('Users ,the last most subscribed')
    col2.write(users_bunble_maxSubs.head(3))

    # Sumilars items recommendation
    recom_content_content=get_bundle_recommendation(a)
    # aggregated with prices
    recom_content_content2= pd.merge(price[['bundle_name','price_point']],recom_content_content,how="right", on='bundle_name')
    # Selects two similar bundles that are superior and closer to our target
    recom_content_content2=recom_content_content2[recom_content_content2['price_point']>price_a]
    recom_content_content2=recom_content_content2.sort_values(by=['price_point'],ascending=True)
    recom_content_content2=pd.DataFrame(recom_content_content2)
    st.write('Recommended')
    st.write(recom_content_content2.head(2))



#######################################################################################################################################
##                                                                                                                                   ##
###                                                     rc_bundle_kmeans                                                            ###
##                                                                                                                                   ##
#######################################################################################################################################

df_data_BTL = cf_final_dataset
df_data_BTL.rename(columns = {'msisdn_id_x': 'msisdn_id'},inplace = True)

df_kmeans_bundle = df_kmeans.merge(df_data_BTL,on = 'msisdn_id',how='left')
def reco(num):
    bundle = []
    t = df_kmeans_bundle[df_kmeans_bundle.msisdn_id==num].reset_index()
    T1 = t.groupby(['labels','bundle_name','price_point'],dropna=False)['msisdn_id'].count().nlargest(2).reset_index()

# we will perform recommendation to customer in labels 0 like this
# customer without bundle subscription will be recommend BTL 4G Bundle 1 Daily 
# customer with bundle BTL 4G Bundle 1 Daily will be recommend  BTL 4G Bundle 14 Weekly 
# customer with bundle BTL 4G Bundle 14 Weekly will be recommend BTL 4G Bundle 7 Weekly (Upselling)
    if T1.labels[0]== 0: # cluster 0
        bundle.append(T1.labels[0])
        if len(T1) == 0 :
            bundle.append('BTL 4G Bundle 1 Daily')
            bundle.append('BTL 4G Bundle 14 Weekly')
        elif len(T1) == 1 :
            if T1.bundle_name[0] == 'BTL 4G Bundle 1 Daily ':
                    bundle.append('BTL 4G Bundle 1 Daily')
                    bundle.append('BTL 4G Bundle 14 Weekly')
            elif T1.bundle_name[0] == 'BTL 4G Bundle 14 Weekly':
                    bundle.append('BTL 4G Bundle 14 Weekly')
                    bundle.append('BTL 4G Bundle 7 Weekly') # uspselling
            else:
                if T1.price_point[0] > 22 : # I want to prevent down - selling
                    bundle.append(T1.bundle_name[0])
                    bundle.append('BTL 4G Bundle 7 Weekly')
                else:
                    bundle.append('BTL 4G Bundle 14 Weekly')
                    bundle.append('BTL 4G Bundle 7 Weekly')
        else:     
            if T1.price_point[1] > 22:  # I want to prevent down - selling
                    bundle.append(T1.bundle_name[0])
                    bundle.append(T1.bundle_name[1])
            else:
                    bundle.append(T1.bundle_name[0])
                    bundle.append('BTL 4G Bundle 7 Weekly')# Upselling

# we will perform recommendation to customer in labels 1 like this
# customer without bundle subscription will be recommend BTL 4G Bundle 14 Weekly
# customer with bundle BTL 4G Bundle 14 Weekly will be recommend BTL 4G Bundle 7 Weekly 
# customer with bundle BTL 4G Bundle 7 Weekly   will be recommend BTL 4G Bundle 8 Weekly(Upselling
    
    
    if T1.labels[0]== 1:   # cluster 1
        bundle.append(T1.labels[0])
        if len(T1) == 0 :
            bundle.append('BTL 4G Bundle 7 Weekly')
            bundle.append('BTL 4G Bundle 14 Weekly')
        elif len(T1) == 1 :
            if T1.bundle_name[0] == 'BTL 4G Bundle 14 Weekly':
                    bundle.append('BTL 4G Bundle 7 Weekly')
                    bundle.append('BTL 4G Bundle 14 Weekly')
            elif T1.bundle_name[0] == 'BTL 4G Bundle 7 Weekly':
                    bundle.append('BTL 4G Bundle 7 Weekly')
                    bundle.append('BTL 4G Bundle 8 Weekly')
            else:
                if T1.price_point[0] > 35 : # I want to prevent down - selling
                    bundle.append(T1.bundle_name[0])
                    bundle.append('BTL 4G Bundle 8 Weekly')
                else:
                    bundle.append('BTL 4G Bundle 7 Weekly')
                    bundle.append('BTL 4G Bundle 8 Weekly')
        else:     
            if T1.price_point[1] > 35:  # I want to prevent down - selling
                    bundle.append(T1.bundle_name[0])
                    bundle.append(T1.bundle_name[1])
            else:
                    bundle.append(T1.bundle_name[0])
                    bundle.append('BTL 4G Bundle 8 Weekly')# Upselling

# we will perform recommendation to customer in labels 2 like this
# customer without bundle subscription will be recommend BTL 4G BTL 4G Bundle 7 Weekly  
# customer with BTL 4G Bundle 7 Weekly  will be recommend BTL 4G Bundle 8 Weekly 
# customer with bundle BTL 4G Bundle 8 Weekly  will be recommend BTL 4G Bundle 9 Monthly (Upselling)                   
     
    if T1.labels[0]== 2:   # cluster 2
        bundle.append(T1.labels[0])
        if len(T1) == 0 :
            bundle.append('BTL 4G Bundle 7 Weekly')
            bundle.append('BTL 4G Bundle 8 Weekly')
        elif len(T1) == 1 :
            if T1.bundle_name[0] == 'BTL 4G Bundle 7 Weekly':
                    bundle.append('BTL 4G Bundle 7 Weekly')
                    bundle.append('BTL 4G Bundle 8 Weekly')
            elif T1.bundle_name[0] == 'BTL 4G Bundle 8 Weekly':
                    bundle.append('BTL 4G Bundle 8 Weekly')
                    bundle.append('BTL 4G Bundle 9 Weekly')
            else:
                if T1.price_point[0] > 50 : # I want to prevent down - selling
                    bundle.append(T1.bundle_name[0])
                    bundle.append('BTL 4G Bundle 9 Weekly')
                else:
                    bundle.append('BTL 4G Bundle 8 Weekly')
                    bundle.append('BTL 4G Bundle 9 Weekly')
        else:     
            if T1.price_point[1] > 50:  # I want to prevent down - selling
                    bundle.append(T1.bundle_name[0])
                    bundle.append(T1.bundle_name[1])
            else:
                    bundle.append(T1.bundle_name[0])
                    bundle.append('BTL 4G Bundle 8 Weekly')# Upselling
                    
    return bundle, st.write(bundle)


st.markdown("### rc_bundle_kmeans ")
# tto select a user for non-subscribers
ac= st.selectbox("Select  user for Kmeanss", pd.unique(df_data_BTL["msisdn_id"]))
#Propose 2 bunble 
if ac:
    df_user_info=df_data_BTL[df_data_BTL["msisdn_id"]==ac]
    user_Nbr_subs=df_user_info.groupby(['msisdn_id'])['sum_subscriptions'].sum()
    user_Nbr_subs=pd.DataFrame(user_Nbr_subs)
    user_Nbr_subs=user_Nbr_subs['sum_subscriptions'].iloc[0]
    users_bunble_maxSubs=df_user_info.sort_values(by=['sum_subscriptions'],ascending=False)
    users_bunble_maxSubs=users_bunble_maxSubs[['bundle_name','sum_subscriptions']]

    user_Nbr_subs=int(user_Nbr_subs)
    col1, col2 = st.columns(2)

    col1.metric(label="Total subscriptions", value=user_Nbr_subs,
        delta_color="inverse")
    col2.write('Users ,the last most subscribed')
    col2.write(users_bunble_maxSubs.head(3))
  
    st.write('Recommended')
    st.write(reco(ac))
