#!/usr/bin/env python3
"""
S02E01 - Analiza nagrań audio z przesłuchań
Obsługuje: openai, lmstudio, anything, gemini, claude
DODANO: Obsługę Claude z bezpośrednią integracją (jak zad1.py i zad2.py)
POPRAWKA: Lepsze wykrywanie silnika z agent.py
POPRAWKA SONARA: Refaktoryzacja funkcji wysokiej złożoności kognitywnej
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

# Konfiguracja i helpery
load_dotenv(override=True)

# Stałe
EXPECTED_AUDIO_EXTENSIONS = [".m4a", ".mp3", ".wav", ".flac"]
ANTHROPIC_INSTALL_MSG = "❌ Musisz zainstalować anthropic: pip install anthropic"
OPENAI_INSTALL_MSG = "❌ Musisz zainstalować openai: pip install openai"
GEMINI_INSTALL_MSG = "❌ Musisz zainstalować google-generativeai: pip install google-generativeai"

# POPRAWKA: Dodano argumenty CLI jak w innych zadaniach
parser = argparse.ArgumentParser(
    description="Analiza audio z przesłuchań (multi-engine + Claude)"
)
parser.add_argument(
    "--engine",
    choices=["openai", "lmstudio", "anything", "gemini", "claude"],
    help="LLM backend to use",
)
args = parser.parse_args()


def detect_engine() -> str:
    """Wykrywa silnik LLM na podstawie argumentów i zmiennych środowiskowych"""
    if args.engine:
        return args.engine.lower()
    elif os.getenv("LLM_ENGINE"):
        return os.getenv("LLM_ENGINE").lower()
    else:
        # Próbuj wykryć silnik na podstawie ustawionych zmiennych MODEL_NAME
        model_name = os.getenv("MODEL_NAME", "")
        if "claude" in model_name.lower():
            return "claude"
        elif "gemini" in model_name.lower():
            return "gemini"
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            return "openai"
        else:
            # Sprawdź które API keys są dostępne
            if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
                return "claude"
            elif os.getenv("GEMINI_API_KEY"):
                return "gemini"
            elif os.getenv("OPENAI_API_KEY"):
                return "openai"
            else:
                return "lmstudio"  # domyślnie


def validate_engine(engine: str) -> None:
    """Waliduje czy silnik jest obsługiwany"""
    if engine not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
        print(f"❌ Nieobsługiwany silnik: {engine}", file=sys.stderr)
        sys.exit(1)


def validate_environment() -> None:
    """Sprawdza czy wszystkie wymagane zmienne środowiskowe są ustawione"""
    required_vars = ["DATA_URL", "REPORT_URL", "CENTRALA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(
            f"❌ Brak wymaganych zmiennych: {', '.join(missing_vars)}",
            file=sys.stderr,
        )
        sys.exit(1)


# Inicjalizacja
ENGINE = detect_engine()
validate_engine(ENGINE)
validate_environment()

print(f"🔄 ENGINE wykryty: {ENGINE}")

DATA_URL = os.getenv("DATA_URL")
REPORT_URL = os.getenv("REPORT_URL")
CENTRALA_KEY = os.getenv("CENTRALA_API_KEY")


# --- KLASY LLM CLIENT - POPRAWKA SONARA S3776 ---
class AudioAnalysisClient(ABC):
    """Bazowa klasa dla klientów analizy audio"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def infer_answer(self, fragments: str) -> str:
        """Metoda do wnioskowania odpowiedzi - implementacja w podklasach"""
        pass
    
    @abstractmethod
    def get_transcript(self, audio_path: Path) -> str:
        """Metoda do transkrypcji audio - implementacja w podklasach"""
        pass


class OpenAIAnalysisClient(AudioAnalysisClient):
    """Klient analizy dla OpenAI"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError:
            print(OPENAI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def get_transcript(self, audio_path: Path) -> str:
        """Pobiera transkrypcję z cache lub generuje nową via Whisper"""
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists():
            print(f"   > Używam zapisanej transkrypcji: {txt_path.name}")
            return txt_path.read_text(encoding="utf-8")

        print(f"   > Transkrypcja z API dla: {audio_path.name}")
        with open(audio_path, "rb") as f:
            resp = self.client.audio.transcriptions.create(
                file=f, model="whisper-1", response_format="text", language="pl"
            )

        text = getattr(resp, "text", resp)
        txt_path.write_text(text, encoding="utf-8")
        return text
    
    def infer_answer(self, fragments: str) -> str:
        system_msg, user_msg = self._create_inference_prompts(fragments)
        
        print(f"[DEBUG] Wysyłam zapytanie do OpenAI z fragmentami zeznań")
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        
        self._log_usage(resp.usage, "openai")
        return answer
    
    def _log_usage(self, usage: Any, engine: str) -> None:
        """Loguje użycie tokenów i koszty"""
        tokens = usage
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        cost = (
            tokens.prompt_tokens / 1_000_000 * 0.60
            + tokens.completion_tokens / 1_000_000 * 2.40
        )
        print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
    
    def _create_inference_prompts(self, fragments: str) -> tuple[str, str]:
        """Tworzy prompty dla wnioskowania"""
        system_msg = (
            "Jesteś śledczym-logiki."
            "Otrzymasz dwa fragmenty zeznań dotyczące przedmiotu i miejsca wykładów."
            '1. Wypisz "Fakt 1" - przedmiot wykładów.'
            '2. Wypisz "Fakt 2" - miasto wykładów.'
            "3. Na podstawie tych faktów wnioskuj nazwę wydziału uczelni."
            "4. Korzystając z wiedzy ogólnej, podaj ulicę siedziby wydziału."
            "Najpierw rozpisz chain-of-thought, potem odpowiedź w formacie:"
            "Wydział: <pełna nazwa>Ulica: <ulica i numer>"
        )
        user_msg = f"Fragmenty zeznań:{fragments}"
        return system_msg, user_msg


class LocalAnalysisClient(AudioAnalysisClient):
    """Klient analizy dla lokalnych modeli (LMStudio, Anything)"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str, engine_name: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError:
            print(OPENAI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self.engine_name = engine_name
        print(f"[DEBUG] {engine_name} URL: {base_url}")
        print(f"[DEBUG] {engine_name} Model: {model_name}")
    
    def get_transcript(self, audio_path: Path) -> str:
        """Pobiera transkrypcję z cache lub generuje nową"""
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists():
            print(f"   > Używam zapisanej transkrypcji: {txt_path.name}")
            return txt_path.read_text(encoding="utf-8")

        print(f"   > Transkrypcja z API dla: {audio_path.name}")
        # Lokalne modele mogą mieć inny endpoint dla audio
        transcribe_url = os.getenv("TRANSCRIBE_API_URL", "http://localhost:1234/v1")
        transcribe_client = OpenAI(
            api_key=self.client.api_key,
            base_url=transcribe_url,
        )
        
        with open(audio_path, "rb") as f:
            resp = transcribe_client.audio.transcriptions.create(
                file=f,
                model="whisper-1",  # lub lokalny model
                response_format="text",
                language="pl",
            )

        text = getattr(resp, "text", resp)
        txt_path.write_text(text, encoding="utf-8")
        return text
    
    def infer_answer(self, fragments: str) -> str:
        system_msg, user_msg = self._create_inference_prompts(fragments)
        
        print(f"[DEBUG] Wysyłam zapytanie do {self.engine_name} z fragmentami zeznań")
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        
        self._log_usage(resp.usage)
        return answer
    
    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów dla lokalnych modeli"""
        tokens = usage
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        print(f"[💰 Model lokalny ({self.engine_name}) - brak kosztów]")
    
    def _create_inference_prompts(self, fragments: str) -> tuple[str, str]:
        """Tworzy prompty dla wnioskowania"""
        system_msg = (
            "Jesteś śledczym-logiki."
            "Otrzymasz dwa fragmenty zeznań dotyczące przedmiotu i miejsca wykładów."
            '1. Wypisz "Fakt 1" - przedmiot wykładów.'
            '2. Wypisz "Fakt 2" - miasto wykładów.'
            "3. Na podstawie tych faktów wnioskuj nazwę wydziału uczelni."
            "4. Korzystając z wiedzy ogólnej, podaj ulicę siedziby wydziału."
            "Najpierw rozpisz chain-of-thought, potem odpowiedź w formacie:"
            "Wydział: <pełna nazwa>Ulica: <ulica i numer>"
        )
        user_msg = f"Fragmenty zeznań:{fragments}"
        return system_msg, user_msg


class ClaudeAnalysisClient(AudioAnalysisClient):
    """Klient analizy dla Claude"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            from anthropic import Anthropic
        except ImportError:
            print(ANTHROPIC_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        self.client = Anthropic(api_key=api_key)
        print(f"[DEBUG] Claude Model: {model_name}")
    
    def get_transcript(self, audio_path: Path) -> str:
        """Claude nie obsługuje transkrypcji audio"""
        print(f"❌ Transkrypcja audio (Whisper) nie jest dostępna dla Claude.")
        print("💡 Użyj --engine openai, lmstudio lub anything dla transkrypcji audio.")
        sys.exit(1)
    
    def infer_answer(self, fragments: str) -> str:
        system_msg, user_msg = self._create_inference_prompts(fragments)
        
        print("[DEBUG] Wysyłam zapytanie do Claude z fragmentami zeznań")
        resp = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": system_msg + "\n\n" + user_msg}],
            temperature=0,
            max_tokens=4000,
        )
        
        self._log_usage(resp.usage)
        return resp.content[0].text.strip()
    
    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów dla Claude"""
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
        print(
            f"[📊 Prompt: {usage.input_tokens} | "
            f"Completion: {usage.output_tokens} | "
            f"Total: {usage.input_tokens + usage.output_tokens}]"
        )
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")
    
    def _create_inference_prompts(self, fragments: str) -> tuple[str, str]:
        """Tworzy prompty dla wnioskowania"""
        system_msg = (
            "Jesteś śledczym-logiki."
            "Otrzymasz dwa fragmenty zeznań dotyczące przedmiotu i miejsca wykładów."
            '1. Wypisz "Fakt 1" - przedmiot wykładów.'
            '2. Wypisz "Fakt 2" - miasto wykładów.'
            "3. Na podstawie tych faktów wnioskuj nazwę wydziału uczelni."
            "4. Korzystając z wiedzy ogólnej, podaj ulicę siedziby wydziału."
            "Najpierw rozpisz chain-of-thought, potem odpowiedź w formacie:"
            "Wydział: <pełna nazwa>Ulica: <ulica i numer>"
        )
        user_msg = f"Fragmenty zeznań:{fragments}"
        return system_msg, user_msg


class GeminiAnalysisClient(AudioAnalysisClient):
    """Klient analizy dla Gemini"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            import google.generativeai as genai
        except ImportError:
            print(GEMINI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"[DEBUG] Gemini Model: {model_name}")
    
    def get_transcript(self, audio_path: Path) -> str:
        """Gemini nie obsługuje transkrypcji audio"""
        print(f"❌ Transkrypcja audio (Whisper) nie jest dostępna dla Gemini.")
        print("💡 Użyj --engine openai, lmstudio lub anything dla transkrypcji audio.")
        sys.exit(1)
    
    def infer_answer(self, fragments: str) -> str:
        system_msg, user_msg = self._create_inference_prompts(fragments)
        
        print("[DEBUG] Wysyłam zapytanie do Gemini z fragmentami zeznań")
        response = self.model.generate_content(
            [system_msg, user_msg],
            generation_config={"temperature": 0.0, "max_output_tokens": 512},
        )
        
        self._log_usage()
        return response.text.strip()
    
    def _log_usage(self) -> None:
        """Loguje informacje o użyciu dla Gemini"""
        print("[📊 Gemini - brak szczegółów tokenów]")
        print("[💰 Gemini - sprawdź limity w Google AI Studio]")
    
    def _create_inference_prompts(self, fragments: str) -> tuple[str, str]:
        """Tworzy prompty dla wnioskowania"""
        system_msg = (
            "Jesteś śledczym-logiki."
            "Otrzymasz dwa fragmenty zeznań dotyczące przedmiotu i miejsca wykładów."
            '1. Wypisz "Fakt 1" - przedmiot wykładów.'
            '2. Wypisz "Fakt 2" - miasto wykładów.'
            "3. Na podstawie tych faktów wnioskuj nazwę wydziału uczelni."
            "4. Korzystając z wiedzy ogólnej, podaj ulicę siedziby wydziału."
            "Najpierw rozpisz chain-of-thought, potem odpowiedź w formacie:"
            "Wydział: <pełna nazwa>Ulica: <ulica i numer>"
        )
        user_msg = f"Fragmenty zeznań:{fragments}"
        return system_msg, user_msg


def create_analysis_client() -> AudioAnalysisClient:
    """Factory function dla tworzenia klienta analizy audio"""
    if ENGINE == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
            sys.exit(1)
        
        base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
        return OpenAIAnalysisClient(model_name, api_key, base_url)
    
    elif ENGINE == "lmstudio":
        api_key = os.getenv("LMSTUDIO_API_KEY", "local")
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
        return LocalAnalysisClient(model_name, api_key, base_url, "LMStudio")
    
    elif ENGINE == "anything":
        api_key = os.getenv("ANYTHING_API_KEY", "local")
        base_url = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")
        return LocalAnalysisClient(model_name, api_key, base_url, "Anything")
    
    elif ENGINE == "claude":
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env", file=sys.stderr)
            sys.exit(1)
        
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
        return ClaudeAnalysisClient(model_name, api_key)
    
    elif ENGINE == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Brak GEMINI_API_KEY w .env", file=sys.stderr)
            sys.exit(1)
        
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
        return GeminiAnalysisClient(model_name, api_key)


# Inicjalizacja globalnego klienta
analysis_client = create_analysis_client()
print(f"✅ Zainicjalizowano silnik: {ENGINE} z modelem: {analysis_client.model_name}")


# --- FUNKCJE POMOCNICZE - POPRAWKA SONARA S3776 ---
def download_and_extract_zip(url: str, dest: Path) -> None:
    """Pobiera i rozpakuje archiwum ZIP"""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "przesluchania.zip"
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()


def find_audio_files(root: Path, exts: Optional[List[str]] = None) -> List[Path]:
    """Znajduje pliki audio w danym katalogu"""
    if exts is None:
        exts = EXPECTED_AUDIO_EXTENSIONS
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def ensure_audio_files_exist(base_dir: Path) -> List[Path]:
    """Zapewnia istnienie plików audio - pobiera jeśli potrzeba"""
    audio_files = find_audio_files(base_dir) if base_dir.exists() else []
    if not audio_files:
        print("1/4 Pobieram i rozpakowuję nagrania...")
        download_and_extract_zip(DATA_URL, base_dir)
        audio_files = find_audio_files(base_dir)
    else:
        print("1/4 Pliki audio już istnieją, pomijam pobieranie i rozpakowywanie.")
    return audio_files


def process_transcripts(audio_files: List[Path]) -> str:
    """Przetwarza transkrypcje audio i łączy je w jeden tekst"""
    transcripts = []
    for audio in audio_files:
        print(f"2/4 Przetwarzanie: {audio.name}")
        text = analysis_client.get_transcript(audio)
        # Pomijamy transkrypcje zawierające "arkadiusz"
        if "arkadiusz" not in text.lower():
            transcripts.append(text)
    return "\n".join(transcripts)


def extract_fragments(combined_text: str) -> str:
    """Ekstraktuje kluczowe fragmenty z połączonego tekstu"""
    subject = None
    location = None
    
    # Szukamy kluczowych informacji w tekście
    for line in combined_text.split("\n"):
        line_lower = line.lower()
        
        if subject is None and ("informatyka" in line_lower or "matematyka" in line_lower):
            subject = line.strip()
        
        if location is None and "krakowie" in line_lower:
            location = line.strip()
        
        if subject and location:
            break
    
    # Zbieramy znalezione fragmenty
    fragments = []
    if subject:
        fragments.append(subject)
    if location:
        fragments.append(location)
    
    # Jeśli nie znaleziono kluczowych fragmentów, weź ostatnie dwie linie
    if not fragments:
        last_two = combined_text.split("\n")[-2:]
        fragments = [line.strip() for line in last_two if line.strip()]
    
    return "\n".join(fragments)


def send_report(answer: str) -> None:
    """Wysyła raport do serwera"""
    print("4/4 Wysyłam raport...")
    payload = {"task": "mp3", "apikey": CENTRALA_KEY, "answer": answer}
    resp = requests.post(REPORT_URL, json=payload)
    
    if resp.status_code == 200:
        print("✅ Odpowiedź wysłana, serwer odpowiedział:", resp.json())
    else:
        print(f"❌ Błąd przy wysyłaniu: {resp.status_code}\n{resp.text}")


def main() -> None:
    """
    POPRAWKA SONARA S3776: Główna funkcja programu
    Refaktoryzacja z wysokiej złożoności kognitywnej na prostą orkiestrację
    """
    print(f"🚀 Używam silnika: {ENGINE}")
    base_dir = Path("przesluchania")

    # 1. Zapewnij istnienie plików audio
    audio_files = ensure_audio_files_exist(base_dir)

    # 2. Przetwórz transkrypcje
    combined_text = process_transcripts(audio_files)

    # 3. Wyciągnij kluczowe fragmenty
    fragments = extract_fragments(combined_text)
    print(f"🔍 Znalezione fragmenty:\n{fragments}")

    # 4. Wnioskowanie z pomocą LLM
    print("3/4 Wnioskuję z pomocą LLM...")
    answer = analysis_client.infer_answer(fragments)
    print(f"Odpowiedź:\n{answer}")

    # 5. Wysłanie raportu
    send_report(answer)


if __name__ == "__main__":
    main()