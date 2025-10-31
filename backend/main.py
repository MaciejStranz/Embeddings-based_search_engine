from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from .search_engine import init_qdrant_client, init_model, initialize_collection, search, insert_data_from_csv
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Initializing model and qdrant client")
    app.state.model = init_model()
    app.state.qdrant = init_qdrant_client()

    if settings.CSV_PATH and initialize_collection(app.state.qdrant, settings.COLLECTION_NAME):
        try:
            print("Inserting data form csv")
            insert_data_from_csv(
                client = app.state.qdrant,
                model = app.state.model,
                collection_name = settings.COLLECTION_NAME,
                path = settings.CSV_PATH
            )
            print("Succesfully inserted data from csv")
        except FileNotFoundError:
            print(f"[startup] CSV not found: {settings.CSV_PATH}")
    else:
        print(f"Collection {settings.COLLECTION_NAME} already exists, so CSV data was not inserted")

    yield 

    print("[shutdown] Shutting aplication down")


app = FastAPI(title="Embeddings Search API", lifespan=lifespan)

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "collection": settings.COLLECTION_NAME,
        "model": settings.MODEL_NAME,
        "csv_loaded": bool(settings.CSV_PATH),
    }

# @app.post("/search", response_model=SearchResponse)
# def search_endpoint(req: SearchRequest, request: Request):
#     if not req.query.strip():
#         raise HTTPException(status_code=400, detail="Empty query")

#     qdrant = request.app.state.qdrant
#     model = request.app.state.model

#     results, took_ms = search(
#         client=qdrant,
#         model=model,
#         collection=settings.QDRANT_COLLECTION,
#         query=req.query,
#         top_k=req.top_k,
#     )

#     hits = [SearchHit(id=r["id"], score=r["score"], payload=r["payload"]) for r in results]
#     return SearchResponse(query=req.query, results=hits, took_ms=took_ms)

