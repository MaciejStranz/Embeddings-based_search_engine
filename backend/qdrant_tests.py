from .search_engine import init_qdrant_client, init_model, initialize_collection, search, insert_data_from_csv
from sentence_transformers import SentenceTransformer

model = init_model()
col_name = "movies"
# csv_path = "movies_reviews.csv"
q = "I am interested in criminal movie about serial killers"

client = init_qdrant_client()
if(initialize_collection(client, col_name)):
    insert_data_from_csv(client, model, col_name)

res = search(client, q, model, 5, col_name)
print(res)
