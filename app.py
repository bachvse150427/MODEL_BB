from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import pandas as pd
from inferencer import Inference

app = FastAPI(
    title="Stock Market Prediction API",
    description="API for making stock market predictions using trained models",
    version="1.0.0"
)

# Initialize inference engine
inferencer = Inference()

class PredictionResponse(BaseModel):
    ticker: str
    date: str
    prediction: int
    prediction_label: str
    model_used: str
    f1_score: float
    confidence: Optional[float] = None
    timestamp: str

class AllTickersResponse(BaseModel):
    results: Dict[str, PredictionResponse]

def get_today_date():
    return date.today().strftime('%Y-%m-%d')

@app.get("/")
async def root():
    return {"message": "Welcome to Stock Market Prediction API"}

@app.get("/predict/ticker/{ticker}", response_model=PredictionResponse)
async def predict_ticker(ticker: str):
    try:
        result = inferencer.infer_last_point(ticker)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"No prediction available for ticker {ticker}")
        
        # Add prediction_label and timestamp
        result['prediction_label'] = '1' if result['Prediction'] == 1 else '0'
        result['timestamp'] = datetime.now().isoformat()
            
        response = {
            'ticker': result.get('Ticker', ticker),
            'date': result.get('Date'),
            'prediction': result.get('Prediction'),
            'prediction_label': result.get('prediction_label'),
            'model_used': result.get('model_used'),
            'f1_score': result.get('f1_score'),
            'timestamp': result.get('timestamp')
        }
        
        if 'confidence' in result:
            response['confidence'] = result['confidence']
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/all-tickers", response_model=AllTickersResponse)
async def predict_all_tickers():
    try:
        results = inferencer.infer_all_last_points()
        
        if not results:
            raise HTTPException(status_code=404, detail="No predictions available")
        
        processed_results = {}
        for ticker, result in results.items():
            # Add prediction_label and timestamp
            result['prediction_label'] = '1' if result['Prediction'] == 1 else '0'
            result['timestamp'] = datetime.now().isoformat()
            
            processed_result = {
                'ticker': result.get('Ticker', ticker),
                'date': result.get('Date'),
                'prediction': result.get('Prediction'),
                'prediction_label': result.get('prediction_label'),
                'model_used': result.get('model_used'),
                'f1_score': result.get('f1_score'),
                'timestamp': result.get('timestamp')
            }
            
            if 'confidence' in result:
                processed_result['confidence'] = result['confidence']
                
            processed_results[ticker] = processed_result
        
        return {"results": processed_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)