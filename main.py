from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import tempfile
from onnx_inference_demo import cli_infer

app = FastAPI()

@app.post("/convert_voice/")
async def process_audio(audio_file: UploadFile = File(...)):    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file_in:
        audio_data_in = await audio_file.read()
        temp_file_in.write(audio_data_in)
        temp_file_in.flush()
        temp_filename_in = temp_file_in.name
    
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file_out:
            temp_file_out.flush()
            temp_filename_out = temp_file_out.name
            cli_infer(f"Spongebob.pth {temp_filename_in} {temp_filename_out} logs/added_IVF6717_Flat_nprobe_1_Spongebob_v2.index 0 10 harvest 120 3 0 1 0.78 0.33 false")

            return FileResponse(temp_filename_out, media_type="audio/wav", status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
