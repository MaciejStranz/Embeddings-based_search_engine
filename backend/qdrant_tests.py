from .search_engine import get_client, initialize_collection, search, insert_data_from_csv, COLLECTION_NAME
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col_name = "movies"
csv_path = "movies_reviews.csv"
q = "I am interested in criminal movie about serial killers"

client = get_client()
if(initialize_collection(client, col_name)):
    insert_data_from_csv(client, model, col_name, csv_path)

res = search(client, q, model, col_name)
print(res)
