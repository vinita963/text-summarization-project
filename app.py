from fastapi import FastAPI
import uvicorn
import os
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(title="Generative AI Summarizer API (Phi-3)")

@app.get("/", tags=["Dashboard"])
async def index():
    return {"message": "Head to /docs to test the API endpoints"}

@app.get("/train", tags=["Training"])
async def training():
    try:
        # Warning: On local laptops, this may crash due to out-of-memory.
        # Run this endpoint in Colab!
        os.system("python main.py")
        return {"message": "Training Pipeline Executed Successfully! LoRA weights saved."}
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict", tags=["Inference"])
async def predict_route(text: str):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return {"summary": summary}
    except Exception as e:
        return Response(f"Error processing text: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
