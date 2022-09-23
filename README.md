# bp_rcom_bundle
Recommandation Bundle

to run in local: 

    streamlit run app.py 

to run on line with heroku : https://rc-bundle.herokuapp.com/

 2 BTL bundles recommende for each customers.

for additional data sources go to this link below: https://drive.google.com/drive/folders/1jKlJcppBPPJqiCvj3u4nSbTAfz8hGrfu?usp=sharing
 
  1-) For Exploratory Analysis open Assessment_Exploratory_analysis.ipynb  file
  
  2-) Kmean clustering and profiling open  clustering profiling.ipynb fil
   
  3) Three way to perform recommandation
      - for unsunbribes custimers we recommend the two top BTL bundle by soubscibtions
      - Kmean recommandatin :  For each clusters we Take the two top bundle and on this we recommend two best BTL bundle base on historical data of the customers in the clusters
      - Collaboratif filtering item-item base : for each customers we look your top bundle and their similarities base on CL algorithms. And depends on the similarities we recommende two best bundle for the customers
   
  

