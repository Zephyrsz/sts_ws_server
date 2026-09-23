import uvicorn

if __name__ == "__main__":
    uvicorn.run("sts_ws_server.app:app", host="0.0.0.0", port=8000, workers=1)
