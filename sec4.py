#!/usr/bin/env python3
"""
Sekretne zadanie z odwróconym audio Vimeo — Wersja uniwersalna!
Obsługuje: openai, lmstudio, anything, gemini, claude
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp
from dotenv import load_dotenv

load_dotenv(override=True)

# Constants
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "lmstudio": "llama-3.3-70b-instruct", 
    "anything": "llama-3.3-70b-instruct",
    "claude": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.5-pro-latest"
}

TRANSLATIONS = {"sky": "niebo"}

# Initialize argument parser
parser = argparse.ArgumentParser(description="Vimeo Audio Processor - sekretne zadanie")
parser.add_argument(
    "--engine",
    choices=["openai", "lmstudio", "anything", "gemini", "claude"],
    help="LLM backend to use",
)
args = parser.parse_args()

def _detect_engine_from_model_name(model_name: str) -> Optional[str]:
    """Detect engine based on model name."""
    model_name_lower = model_name.lower()
    if "claude" in model_name_lower:
        return "claude"
    elif "gemini" in model_name_lower:
        return "gemini"
    elif "gpt" in model_name_lower or "openai" in model_name_lower:
        return "openai"
    return None

def _detect_engine_from_api_keys() -> str:
    """Detect engine based on available API keys."""
    if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    elif os.getenv("GEMINI_API_KEY"):
        return "gemini"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    else:
        return "lmstudio"

def detect_engine() -> str:
    """Detect the LLM engine to use based on arguments and environment variables."""
    # Priority 1: Explicit argument
    if args.engine:
        return args.engine.lower()
    
    # Priority 2: Environment variable
    if os.getenv("LLM_ENGINE"):
        return os.getenv("LLM_ENGINE").lower()
    
    # Priority 3: Model name inference
    model_name = os.getenv("MODEL_NAME", "")
    if model_name:
        detected = _detect_engine_from_model_name(model_name)
        if detected:
            return detected
    
    # Priority 4: API key detection
    return _detect_engine_from_api_keys()

ENGINE = detect_engine()

if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"❌ Nieobsługiwany silnik: {ENGINE}", file=sys.stderr)
    sys.exit(1)

print(f"🔄 ENGINE wykryty: {ENGINE}")

VIMEO_URL = os.getenv("VIMEO_URL")
if not VIMEO_URL:
    print("❌ Brak VIMEO_URL w .env", file=sys.stderr)
    sys.exit(1)


class VimeoExtractor:
    def __init__(self, download_path="./downloads"):
        self.download_path = Path(download_path)
        self.download_path.mkdir(exist_ok=True)

    def download_audio_only(self, url: str) -> Optional[Path]:
        """Download audio from Vimeo URL."""
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.download_path / "%(title)s_audio.%(ext)s"),
            "writeinfojson": True,
            "writethumbnail": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
                print(f"✅ Audio extracted to {self.download_path}")
                audio_files = list(self.download_path.glob("*_audio.mp3"))
                if audio_files:
                    return audio_files[-1]
                else:
                    print("❌ No audio file found after download")
                    return None
        except Exception as e:
            print(f"❌ Error downloading audio: {e}")
            return None


class AudioProcessor:
    def __init__(self, engine="openai"):
        self.engine = engine.lower()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="audio_proc_"))
        self.setup_llm_client()

    def _get_model_name(self) -> str:
        """Get model name based on engine and environment variables."""
        return (os.getenv("MODEL_NAME") or 
                os.getenv(f"MODEL_NAME_{self.engine.upper()}", 
                          DEFAULT_MODELS.get(self.engine, "")))

    def _setup_openai_client(self):
        """Setup OpenAI client."""
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY")
        
        api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=api_key, base_url=api_url)

    def _setup_local_client(self, engine_name: str):
        """Setup local client (LMStudio or Anything LLM)."""
        from openai import OpenAI
        
        api_key = os.getenv(f"{engine_name.upper()}_API_KEY", "local")
        api_url = os.getenv(f"{engine_name.upper()}_API_URL", "http://localhost:1234/v1")
        self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=120)

    def _setup_claude_client(self):
        """Setup Claude client."""
        from anthropic import Anthropic
        
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing CLAUDE_API_KEY or ANTHROPIC_API_KEY")
        
        self.claude_client = Anthropic(api_key=api_key)

    def _setup_gemini_client(self):
        """Setup Gemini client."""
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GEMINI_API_KEY")
        
        genai.configure(api_key=api_key)
        self.model_gemini = genai.GenerativeModel(self.model_name)

    def setup_llm_client(self):
        """Setup LLM client based on engine."""
        self.model_name = self._get_model_name()
        
        if self.engine == "openai":
            self._setup_openai_client()
        elif self.engine == "lmstudio":
            self._setup_local_client("lmstudio")
        elif self.engine == "anything":
            self._setup_local_client("anything")
        elif self.engine == "claude":
            self._setup_claude_client()
        elif self.engine == "gemini":
            self._setup_gemini_client()
        else:
            raise ValueError(f"❌ Unsupported engine: {self.engine}")
        
        print(f"✅ Initialized {self.engine} with model: {self.model_name}")

    def extract_audio_segment(
        self, audio_path: Path, start_time: float, end_time: Optional[float] = None
    ) -> Optional[Path]:
        """Extract audio segment from the main audio file."""
        output_path = self.temp_dir / f"segment_{start_time}s.mp3"
        
        try:
            cmd = self._build_ffmpeg_extract_command(audio_path, start_time, end_time, output_path)
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ Extracted segment: {start_time}s to {end_time or 'end'}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error extracting segment: {e}")
            return None

    def _build_ffmpeg_extract_command(self, audio_path: Path, start_time: float, 
                                     end_time: Optional[float], output_path: Path) -> list:
        """Build ffmpeg command for extracting audio segment."""
        cmd = ["ffmpeg", "-i", str(audio_path), "-ss", str(start_time)]
        if end_time:
            cmd.extend(["-t", str(end_time - start_time)])
        cmd.extend(["-acodec", "mp3", "-ar", "16000", "-ac", "1", "-y", str(output_path)])
        return cmd

    def reverse_audio(self, audio_path: Path) -> Optional[Path]:
        """Reverse the audio file."""
        output_path = self.temp_dir / f"reversed_{audio_path.name}"
        cmd = [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-af",
            "areverse",
            "-y",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ Reversed audio: {output_path.name}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error reversing audio: {e}")
            return None

    def _transcribe_with_whisper_local(self, audio_path: Path, language: str) -> str:
        """Transcribe audio using local whisper."""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path), language=language)
            text = result["text"].strip()
            print(f"✅ Lokalna transkrypcja whisper: {text}")
            return text
        except Exception as e:
            print(f"❌ Błąd lokalnej transkrypcji whisper: {e}")
            return ""

    def _transcribe_with_openai(self, audio_path: Path, language: str) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            with open(audio_path, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    file=f,
                    model="whisper-1",
                    response_format="text",
                    language=language,
                )
                text = getattr(response, "text", response)
                print(f"✅ Transcription completed: {len(text)} characters")
                return text.strip()
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

    def _transcribe_with_openai_fallback(self, audio_path: Path, language: str) -> str:
        """Fallback transcription using OpenAI API."""
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Brak transkrypcji — brak klucza OpenAI.")
            return ""
        
        print("🔄 Fallback: Using OpenAI Whisper for transcription...")
        try:
            from openai import OpenAI
            
            fallback_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1"),
            )
            with open(audio_path, "rb") as f:
                response = fallback_client.audio.transcriptions.create(
                    file=f,
                    model="whisper-1",
                    response_format="text",
                    language=language,
                )
            text = getattr(response, "text", response).strip()
            print(f"✅ Fallback transcription completed: {len(text)} characters")
            return text
        except Exception as e:
            print(f"❌ Fallback transcription failed: {e}")
            return ""

    def transcribe_audio(self, audio_path: Path, language="fr") -> str:
        """Transcribe audio file to text."""
        print(f"🎤 Transcribing audio ({language})...")
        
        if self.engine in ["lmstudio", "anything"]:
            print("⚠️ LM Studio / Anything LLM NIE obsługuje transkrypcji audio (Whisper API).")
            print("🔄 Przełączam na lokalny openai-whisper...")
            return self._transcribe_with_whisper_local(audio_path, language)
        elif self.engine == "openai":
            return self._transcribe_with_openai(audio_path, language)
        elif self.engine in ["claude", "gemini"]:
            print(f"❌ Whisper transcription not available for {self.engine} engine")
            return self._transcribe_with_openai_fallback(audio_path, language)
        else:
            print(f"❌ Unknown engine: {self.engine}")
            return ""

    def _analyze_with_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """Analyze text using OpenAI-compatible API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def _analyze_with_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Analyze text using Claude API."""
        response = self.claude_client.messages.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        return response.content[0].text.strip()

    def _analyze_with_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Analyze text using Gemini API."""
        response = self.model_gemini.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": 0.1, "max_output_tokens": 512},
        )
        return response.text.strip()

    def translate_and_analyze(self, french_text: str) -> str:
        """Translate and analyze French text to extract key information."""
        if not french_text:
            return "❌ No text to analyze"
        
        system_prompt = (
            "You will receive a French sentence. "
            "Respond ONLY with the most essential English word or phrase that describes what the sentence refers to. "
            "If it's about the blue thing above your head, respond with 'sky'. "
            "If it's about a flag, respond with 'flag'. "
            "If possible, respond with just ONE word (in English)."
        )
        user_prompt = french_text.strip()
        print(f"🤖 Analyzing with {self.engine}...")
        print(f"LLM USER PROMPT: {user_prompt!r}")
        
        try:
            if self.engine in ["openai", "lmstudio", "anything"]:
                return self._analyze_with_openai_compatible(system_prompt, user_prompt)
            elif self.engine == "claude":
                return self._analyze_with_claude(system_prompt, user_prompt)
            elif self.engine == "gemini":
                return self._analyze_with_gemini(system_prompt, user_prompt)
        except Exception as e:
            print(f"❌ LLM analysis error: {e}")
            return f"Error during analysis: {e}"

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except (OSError, ImportError):
            pass


class IntegratedProcessor:
    def __init__(self, engine="openai", download_dir="./vimeo_downloads"):
        self.vimeo_extractor = VimeoExtractor(download_dir)
        self.audio_processor = AudioProcessor(engine)

    def _is_valid_transcription(self, transcription: str) -> bool:
        """Check if transcription is valid."""
        return (transcription and 
                not transcription.strip().startswith('{"error"') and
                transcription.strip().lower() not in {"none", ""})

    def _extract_and_reverse_audio(self, audio_path: Path, start_time: float, end_time: float) -> Optional[Path]:
        """Extract audio segment and reverse it."""
        segment_path = self.audio_processor.extract_audio_segment(audio_path, start_time, end_time)
        if not segment_path:
            return None
        return self.audio_processor.reverse_audio(segment_path)

    def process_vimeo_url(self, start_time: float = 58, end_time: float = 61) -> str:
        """Process Vimeo URL and extract flag information."""
        print("🚀 Starting Vimeo Audio Processing Pipeline")
        print("=" * 50)
        url = VIMEO_URL
        print(f"📺 Processing URL from .env: {url}")
        
        print("1/4 📥 Downloading audio from Vimeo...")
        audio_path = self.vimeo_extractor.download_audio_only(url)
        if not audio_path:
            return "❌ Failed to download audio from Vimeo"
        
        print("2/4 🔄 Processing reversed French segment...")
        reversed_path = self._extract_and_reverse_audio(audio_path, start_time, end_time)
        if not reversed_path:
            return "❌ Failed to process audio segment"
        
        print("3/4 🎤 Transcribing audio (fr)...")
        transcription = self.audio_processor.transcribe_audio(reversed_path, language="fr")
        print(f"🇫🇷 French transcription: {transcription}")
        
        if not self._is_valid_transcription(transcription):
            print("❌ Brak poprawnej transkrypcji! Przerywam pipeline.")
            return "❌ Failed to transcribe audio"
        
        fraza = transcription.strip()
        analysis = self.audio_processor.translate_and_analyze(fraza)
        
        print("3/4 🔍 Extracting key information...")
        flag_content = self.extract_flag_content(analysis)
        polska_flaga = TRANSLATIONS.get(flag_content.lower(), flag_content)
        
        print("4/4 🎯 Formatting final result...")
        return f"{{{{FLG:{polska_flaga.upper()}}}}}"

    def _check_simple_word(self, text: str) -> Optional[str]:
        """Check if text is a simple word without spaces."""
        clean_text = text.replace('"', "").replace("'", "").replace(".", "").strip()
        if len(clean_text) <= 20 and " " not in clean_text:
            return clean_text
        return None

    def _extract_from_lines(self, text: str) -> Optional[str]:
        """Extract key word from text lines."""
        lines = text.split("\n")
        for line in lines:
            line = line.strip().lower()
            if not line or line.startswith(("1.", "2.", "3.", "translation:", "answer:", "key:", "response:")):
                continue
            
            words = line.split()
            for word in words:
                word = word.strip(".,!?\"'").lower()
                if word in ["sky", "niebo", "ciel", "himmel", "blue", "niebieski"]:
                    return word
                if 3 <= len(word) <= 15:
                    return word
        return None

    def _extract_from_words(self, text: str) -> Optional[str]:
        """Extract key word from text words."""
        words = text.split()
        excluded_words = {"the", "and", "but", "for", "with", "this", "that"}
        
        for word in words:
            word = word.strip(".,!?\"'()[]{}")
            if len(word) >= 3 and word.lower() not in excluded_words:
                return word
        return None

    def extract_flag_content(self, analysis_text: str) -> str:
        """Extract flag content from analysis text."""
        text = analysis_text.strip()
        
        # Check for simple word
        simple_word = self._check_simple_word(text)
        if simple_word:
            return simple_word
        
        # Extract from lines
        line_result = self._extract_from_lines(text)
        if line_result:
            return line_result
        
        # Extract from words
        word_result = self._extract_from_words(text)
        if word_result:
            return word_result
        
        # Fallback
        return text[:15].strip()

    def cleanup(self):
        """Clean up temporary files."""
        self.audio_processor.cleanup()


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Install with: sudo apt install ffmpeg")
        sys.exit(1)


def main():
    """Main function."""
    check_ffmpeg()
    
    try:
        processor = IntegratedProcessor(ENGINE, "./vimeo_downloads")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    try:
        result = processor.process_vimeo_url(start_time=58, end_time=61)
        print("\n🎯 FINAL RESULT:")
        print("=" * 50)
        print(result)
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()