#!/usr/bin/env python3
"""
S01E05 - Cenzura danych agentów przez LLM
Cenzuruje imię i nazwisko, wiek, miasto oraz ulicę+numer,
zastępując je słowem "CENZURA" wyłącznie przez LLM.
Obsługa: openai, lmstudio, anything, gemini, claude.
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
POPRAWKA: Lepsze wykrywanie silnika z agent.py
POPRAWKA SONARA: Refaktoryzacja funkcji wysokiej złożoności kognitywnej
"""

import argparse
import os
import re
import sys
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# Stałe dla komunikatów błędów
MISSING_OPENAI_KEY_MSG = "❌ Brak OPENAI_API_KEY"
MISSING_CLAUDE_KEY_MSG = "❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env"
MISSING_GEMINI_KEY_MSG = "❌ Brak GEMINI_API_KEY w .env"
UNSUPPORTED_ENGINE_MSG = "❌ Nieobsługiwany silnik:"
MISSING_OPENAI_INSTALL_MSG = "❌ Musisz zainstalować openai: pip install openai"
MISSING_ANTHROPIC_INSTALL_MSG = "❌ Musisz zainstalować anthropic: pip install anthropic"
MISSING_GEMINI_INSTALL_MSG = "❌ Musisz zainstalować google-generativeai: pip install google-generativeai"

# POPRAWKA: Dodano argumenty CLI jak w innych zadaniach
parser = argparse.ArgumentParser(description="Cenzura danych (multi-engine + Claude)")
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
        print(f"{UNSUPPORTED_ENGINE_MSG} {engine}", file=sys.stderr)
        sys.exit(1)


def validate_environment() -> None:
    """Sprawdza czy wszystkie wymagane zmienne środowiskowe są ustawione"""
    required_vars = ["CENTRALA_API_KEY", "REPORT_URL", "CENZURA_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Brak ustawienia: {', '.join(missing_vars)} w .env", file=sys.stderr)
        sys.exit(1)


# Inicjalizacja i walidacja
ENGINE = detect_engine()
validate_engine(ENGINE)
validate_environment()

print(f"🔄 ENGINE wykryty: {ENGINE}")
print(f"✅ Engine: {ENGINE}")

CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
REPORT_URL = os.getenv("REPORT_URL")
CENZURA_URL = os.getenv("CENZURA_URL")

# --- ULTRA-TWARDY PROMPT ---
PROMPT_SYSTEM = (
    "Jesteś automatem do cenzury danych osobowych w języku polskim. "
    "NIE WOLNO Ci zmieniać żadnych innych słów, znaków interpunkcyjnych, układu tekstu ani zamieniać kolejności zdań. "
    "Zamień TYLKO i WYŁĄCZNIE:\n"
    "- każde imię i nazwisko na 'CENZURA',\n"
    "- każdą nazwę miasta na 'CENZURA',\n"
    "- każdą nazwę ulicy wraz z numerem domu/mieszkania na 'CENZURA',\n"
    "- każdą informację o wieku (np. '45 lat', 'wiek: 32', 'lat 27', 'ma 29 lat') na 'CENZURA'.\n"
    "Nie wolno parafrazować, nie wolno podsumowywać, nie wolno streszczać ani zamieniać kolejności czegokolwiek. "
    "Wynikowy tekst musi mieć identyczny układ, interpunkcję i liczbę linii jak oryginał. "
    "Każda inna zmiana niż cenzura wyżej powoduje błąd i NIEZALICZENIE zadania. "
    "Nie pisz żadnych komentarzy, nie wyjaśniaj odpowiedzi. "
    "ODPOWIEDZ WYŁĄCZNIE TEKSTEM Z OCENZURĄ. "
    "PRZYKŁAD:\n"
    "Oryginał:\n"
    "Dane podejrzanego: Jan Kowalski, lat 45, mieszka w Krakowie, ul. Polna 8.\n"
    "Wyjście:\n"
    "Dane podejrzanego: CENZURA, lat CENZURA, mieszka w CENZURA, ul. CENZURA."
)


def download_text(url: str) -> str:
    """Pobiera tekst z podanego URL"""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text.strip()
    except requests.RequestException as e:
        print(f"❌ Błąd podczas pobierania danych: {e}", file=sys.stderr)
        sys.exit(1)


# --- KLASY LLM CLIENT - POPRAWKA SONARA S3776 ---
class LLMCensorClient(ABC):
    """Bazowa klasa dla klientów cenzury LLM"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def censor_text(self, text: str) -> str:
        """Metoda do cenzury tekstu - implementacja w podklasach"""
        pass
    
    def create_user_prompt(self, text: str) -> str:
        """Tworzy prompt użytkownika"""
        return (
            "Tekst do cenzury (nie zmieniaj nic poza danymi osobowymi, przykład wyżej!):\n"
            + text
        )


class OpenAICensorClient(LLMCensorClient):
    """Klient cenzury dla OpenAI"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError:
            print(MISSING_OPENAI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)
        
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0,
        )
        
        self._log_usage(resp.usage)
        return resp.choices[0].message.content.strip()
    
    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów i koszty dla OpenAI"""
        tokens = usage
        cost = (
            tokens.prompt_tokens / 1_000_000 * 0.60
            + tokens.completion_tokens / 1_000_000 * 2.40
        )
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")


class ClaudeCensorClient(LLMCensorClient):
    """Klient cenzury dla Claude"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            from anthropic import Anthropic
        except ImportError:
            print(MISSING_ANTHROPIC_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        self.client = Anthropic(api_key=api_key)
    
    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)
        
        resp = self.client.messages.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": PROMPT_SYSTEM + "\n\n" + prompt_user}
            ],
            temperature=0,
            max_tokens=4000,
        )
        
        self._log_usage(resp.usage)
        return resp.content[0].text.strip()
    
    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów i koszty dla Claude"""
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
        print(
            f"[📊 Prompt: {usage.input_tokens} | "
            f"Completion: {usage.output_tokens} | "
            f"Total: {usage.input_tokens + usage.output_tokens}]"
        )
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")


class GeminiCensorClient(LLMCensorClient):
    """Klient cenzury dla Gemini"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            import google.generativeai as genai
        except ImportError:
            print(MISSING_GEMINI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)
        
        response = self.model.generate_content(
            [PROMPT_SYSTEM + "\n" + prompt_user],
            generation_config={"temperature": 0.0, "max_output_tokens": 4096},
        )
        
        self._log_usage()
        return response.text.strip()
    
    def _log_usage(self) -> None:
        """Loguje informacje o użyciu dla Gemini"""
        print("[📊 Gemini - brak szczegółów tokenów]")
        print("[💰 Gemini - sprawdź limity w Google AI Studio]")


class LocalLLMCensorClient(LLMCensorClient):
    """Klient cenzury dla lokalnych modeli (LMStudio, Anything)"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str, engine_name: str):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError:
            print(MISSING_OPENAI_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.engine_name = engine_name
    
    def censor_text(self, text: str) -> str:
        prompt_user = self.create_user_prompt(text)
        
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0,
        )
        
        self._log_usage(resp.usage)
        return resp.choices[0].message.content.strip()
    
    def _log_usage(self, usage: Any) -> None:
        """Loguje użycie tokenów dla lokalnych modeli"""
        tokens = usage
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        print("[💰 Model lokalny - brak kosztów]")


def create_censor_client() -> LLMCensorClient:
    """Factory function dla tworzenia klienta cenzury LLM"""
    if ENGINE == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(MISSING_OPENAI_KEY_MSG, file=sys.stderr)
            sys.exit(1)
        
        base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
        return OpenAICensorClient(model_name, api_key, base_url)
    
    elif ENGINE == "claude":
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(MISSING_CLAUDE_KEY_MSG, file=sys.stderr)
            sys.exit(1)
        
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
        return ClaudeCensorClient(model_name, api_key)
    
    elif ENGINE == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(MISSING_GEMINI_KEY_MSG, file=sys.stderr)
            sys.exit(1)
        
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
        return GeminiCensorClient(model_name, api_key)
    
    elif ENGINE == "lmstudio":
        api_key = os.getenv("LMSTUDIO_API_KEY", "local")
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct")
        return LocalLLMCensorClient(model_name, api_key, base_url, "LMStudio")
    
    elif ENGINE == "anything":
        api_key = os.getenv("ANYTHING_API_KEY", "local")
        base_url = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct")
        return LocalLLMCensorClient(model_name, api_key, base_url, "Anything")
    
    else:
        print(f"❌ Nieznany silnik: {ENGINE}", file=sys.stderr)
        sys.exit(1)


def censor_llm(text: str) -> str:
    """
    POPRAWKA SONARA S3776: Cenzuruje tekst używając LLM
    Refaktoryzacja z wysokiej złożoności kognitywnej (28) na prostą delegację
    """
    client = create_censor_client()
    return client.censor_text(text)


def extract_flag(text: str) -> str:
    """Wyciąga flagę z tekstu"""
    flag_match = re.search(r"\{\{FLG:[^}]+\}\}|FLG\{[^}]+\}", text)
    return flag_match.group(0) if flag_match else ""


def send_result(censored_text: str) -> None:
    """Wysyła ocenzurowany tekst do serwera"""
    payload = {"task": "CENZURA", "apikey": CENTRALA_API_KEY, "answer": censored_text}
    
    try:
        response = requests.post(REPORT_URL, json=payload, timeout=10)
        if response.ok:
            resp_text = response.text.strip()
            flag = extract_flag(resp_text) or extract_flag(censored_text)
            if flag:
                print(flag)
            else:
                print("Brak flagi w odpowiedzi serwera. Odpowiedź:", resp_text)
        else:
            print(f"❌ Błąd HTTP {response.status_code}: {response.text}", file=sys.stderr)
    except requests.RequestException as e:
        print(f"❌ Błąd podczas wysyłania danych: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Główna funkcja programu"""
    raw_text = download_text(CENZURA_URL)
    print(f"🔄 Pobrano tekst ({len(raw_text)} znaków)")
    print(f"🔄 Cenzuruję używając {ENGINE}...")

    censored_text = censor_llm(raw_text)
    print("=== OCENZUROWANY OUTPUT ===")
    print(censored_text)
    print("===========================")

    send_result(censored_text)


if __name__ == "__main__":
    main()