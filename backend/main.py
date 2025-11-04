from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from .search_engine import init_qdrant_client, init_model, initialize_collection, search, insert_data_from_csv, insert_doc
from .config import settings
from .schemas import SearchResponse, SearchRequest, SearchHit, AddDocRequest, AddDocResponse


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

@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest, request: Request):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    qdrant = request.app.state.qdrant
    model = request.app.state.model
    collection_name = req.collection or settings.COLLECTION_NAME


    results = search(
        client = qdrant,
        query = req.query,
        model = model,
        top_k=req.top_k,
        collection_name = collection_name
    )

    hits = [SearchHit(id=r["id"], score=r["score"], payload=r["payload"]) for r in results]
    return SearchResponse(query=req.query, results=hits)

@app.post("/insert_doc", response_model=AddDocResponse)
def add_document(req: AddDocRequest, request: Request):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    qdrant = request.app.state.qdrant
    model = request.app.state.model
    collection_name = req.collection or settings.COLLECTION_NAME

    try:
        new_id = insert_doc(
            client=qdrant,
            model=model,
            text=req.text.strip(),
            collection_name=collection_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert document: {e}")

    return AddDocResponse(
        id=new_id,
        collection=collection_name,
        payload={"review": req.text.strip()},
    )


