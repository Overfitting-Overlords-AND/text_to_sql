import sentenceCompleter
import fastapi

app = fastapi.FastAPI()

# run with - uvicorn server:app --host 0.0.0.0 --port 8080 --reload
@app.on_event("startup")
async def startup_event():
  print("Starting Up")

@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")

@app.get("/")
def on_root():
  return { "message": "Hello App" }

@app.post("/from_txt_to_sql")
async def text_to_sql(request: fastapi.Request):
  payload = await request.json()
  answer = sentenceCompleter.generate(payload["txt"], payload["cxt"])
  print("Input txt: ", payload["txt"])
  print("Input cxt: ", payload["cxt"])
  return answer
