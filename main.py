from fastapi import FastAPI, Form
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI()

# load model 
MNB = joblib.load('model/MNB.sav')
tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.sav')

class TextData(BaseModel):
    text: str

def preprocess_text(text):
    return text

@app.get("/")
async def check():
    return {"message": "sentiment API"}

@app.post("/predict")
async def predict(text: str = Form(...)):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction = MNB.predict(text_tfidf)
    return {'sentiment': prediction[0]}


# @app.post("/predict")
# async def predict(data: TextData):
#     text = data.text
#     preprocessed_text = preprocess_text(text)
#     text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
#     prediction = MNB.predict(text_tfidf)
#     return {'sentiment': prediction[0]}

if __name__ == '__main__':
    uvicorn.run(app, port=8000)
