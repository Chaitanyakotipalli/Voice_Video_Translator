import os
import tempfile
import io
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import whisper
from deep_translator import GoogleTranslator
from pydantic import BaseModel
import logging
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Translation and Subtitle API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once at startup
stt_model = whisper.load_model("small")  # You can use "base" for faster processing

# Language mapping for TTS
LANG_MAP = {
    "zh": "zh-CN", "ar": "ar", "te": "te", "hi": "hi", "en": "en", 
    "fr": "fr", "de": "de", "it": "it", "ja": "ja", "ko": "ko", 
    "pt": "pt", "ru": "ru", "es": "es", "ta": "ta"
}

class Segment(BaseModel):
    text: str
    start: float
    end: float

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    segments: Optional[List[Segment]] = None

def convert_to_wav(audio_path: str) -> str:
    """Convert audio to WAV format suitable for speech recognition."""
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext != ".wav":
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        temp_wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_wav_fd)
        sound.export(wav_path, format="wav")
        return wav_path
    return audio_path

def transcribe_audio_google(audio_path: str) -> str:
    """Transcribe Telugu audio using Google Speech Recognition."""
    wav_path = convert_to_wav(audio_path)
    is_temp_file = wav_path != audio_path
    
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data, language="te-IN")
        return transcription
    except (sr.RequestError, sr.UnknownValueError) as e:
        logger.warning(f"Google Speech Recognition error: {str(e)}")
        return ""
    finally:
        if is_temp_file and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception as e:
                logger.error(f"Error removing temp file: {str(e)}")

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio using appropriate service based on language."""
    if language == "te":
        return transcribe_audio_google(audio_path)
    
    try:
        options = {"fp16": False, "task": "transcribe"}
        if language and language != "auto":
            options["language"] = language
        
        result = stt_model.transcribe(audio_path, **options)
        return result.get("text", "")
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return ""

def transcribe_audio_with_timestamps(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio using Whisper with timestamp information."""
    try:
        # Set appropriate options
        options = {
            "fp16": False,
            "task": "transcribe",
            "verbose": False
        }
        
        if language and language != "auto":
            options["language"] = language
        
        # Use Whisper to transcribe with word-level timestamps
        result = stt_model.transcribe(
            audio_path, 
            word_timestamps=True,  # Enable word timestamps
            **options
        )
        
        # Extract segments with timestamps
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"]
            })
        
        return {
            "transcription": result.get("text", ""),
            "segments": segments,
            "detected_language": result.get("language", "en")
        }
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using Google Translator."""
    if not text or source_lang == target_lang or (source_lang == "auto" and target_lang == "en"):
        return text
    
    try:
        src = source_lang if source_lang != "auto" else "auto"
        result = GoogleTranslator(source=src, target=target_lang).translate(text)
        if result:
            return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Translation failed")

def translate_text_with_segments(
    text: str, 
    source_lang: str, 
    target_lang: str, 
    segments: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Translate text and preserve segment timestamps."""
    if not text or source_lang == target_lang:
        # No translation needed
        return {
            "translated_text": text,
            "segments": segments
        }
    
    try:
        src = source_lang if source_lang != "auto" else "auto"
        
        # Translate the full text
        translated_full_text = GoogleTranslator(source=src, target=target_lang).translate(text)
        
        # If no segments provided, just return the translated text
        if not segments:
            return {
                "translated_text": translated_full_text,
                "segments": []
            }
        
        # For segments, we need to translate each segment individually to maintain alignment
        translated_segments = []
        for segment in segments:
            segment_text = segment["text"]
            translated_segment = GoogleTranslator(source=src, target=target_lang).translate(segment_text)
            
            translated_segments.append({
                "text": translated_segment,
                "start": segment["start"],
                "end": segment["end"]
            })
        
        return {
            "translated_text": translated_full_text,
            "segments": translated_segments
        }
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def generate_speech(text: str, lang: str = "en") -> bytes:
    """Generate speech from text."""
    tts_lang = LANG_MAP.get(lang, "en")
    try:
        speech = gTTS(text=text, lang=tts_lang, slow=False)
        audio_io = io.BytesIO()
        speech.write_to_fp(audio_io)
        audio_io.seek(0)
        return audio_io.getvalue()
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Audio Translation and Subtitle API",
        "endpoints": [
            "/upload-audio/", 
            "/translate/", 
            "/text-to-speech/", 
            "/speech-to-translation/",
            "/upload-audio-video/",
            "/translate-video/"
        ],
        "supported_languages": {
            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
            "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
            "zh": "Chinese", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
            "ar": "Arabic", "ko": "Korean"
        }
    }

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    language: str = Query(None, description="ISO language code")
):
    """Upload and transcribe audio file."""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
        if not file_ext.startswith('.'):
            file_ext = '.' + file_ext
        
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_filename = tmp.name
        
        transcription_text = transcribe_audio(temp_filename, language)
        
        # Clean up the temp file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        
        if not transcription_text:
            return JSONResponse(
                status_code=422,
                content={"error": "Could not transcribe audio"}
            )
            
        return {"transcription": transcription_text}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload-audio-video/")
async def upload_audio_video(
    file: UploadFile = File(...),
    language: str = Query(None, description="ISO language code"),
    with_timestamps: bool = Query(True, description="Include timestamp data")
):
    """Upload and transcribe audio file with timestamps."""
    try:
        # Save uploaded file to temp location
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
        if not file_ext.startswith('.'):
            file_ext = '.' + file_ext
        
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_filename = tmp.name
        
        # Transcribe audio with timestamp information
        result = transcribe_audio_with_timestamps(temp_filename, language)
        
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        
        return result
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/translate/")
async def translate_text_api(request: TranslationRequest):
    """Translate text from source to target language."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        translated_text = translate_text(
            request.text, 
            request.source_lang, 
            request.target_lang
        )
        return {"translated_text": translated_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/translate-video/")
async def translate_video_api(request: TranslationRequest):
    """Translate text and preserve segment timestamps."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        # Convert Pydantic models to dictionaries for the translation function
        segments = None
        if request.segments:
            segments = [segment.dict() for segment in request.segments]
        
        result = translate_text_with_segments(
            request.text,
            request.source_lang,
            request.target_lang,
            segments
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/text-to-speech/")
async def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    lang: str = Query("en", description="Language code")
):
    """Convert text to speech audio."""
    audio_data = generate_speech(text, lang)
    return Response(content=audio_data, media_type="audio/mp3")

@app.post("/speech-to-translation/")
async def speech_to_translation(
    file: UploadFile = File(...),
    source_lang: str = Query(None, description="Source language"),
    target_lang: str = Query("en", description="Target language")
):
    """Combined endpoint: Speech-to-Text + Translation + TTS."""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
        if not file_ext.startswith('.'):
            file_ext = '.' + file_ext
            
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_filename = tmp.name
        
        transcribed_text = transcribe_audio(temp_filename, source_lang)
        
        # Clean up the temp file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        
        if not transcribed_text:
            return JSONResponse(
                status_code=422,
                content={"error": "Transcription failed"}
            )
        
        translated_text = translate_text(transcribed_text, source_lang or "auto", target_lang)
        audio_data = generate_speech(translated_text, target_lang)
        
        import base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "original_text": transcribed_text,
            "translated_text": translated_text,
            "audio_data_base64": audio_base64
        }
    except Exception as e:
        logger.error(f"Speech-to-translation error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)