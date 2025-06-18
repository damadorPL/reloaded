#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S02E02 (multi-engine + Claude)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• zgodny z openai-python ≥ 1.0
• obsługuje backendy: openai / lmstudio / anything (LocalAI) / gemini (Google) / claude (Anthropic)
• nadal celowo „kłamie" (Kraków, 69, 1999, blue) - logika oryginału zachowana.
DODANO: Obsługę Claude z kompatybilnym interfejsem
POPRAWKA SONARA: Poprawa F-string issues i refaktoryzacja
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from typing import Dict, Tuple, Optional, Any

import requests
import urllib3
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import Claude integration (opcjonalny)
try:
    from claude_integration import (add_token_counting_to_openai_call,
                                    setup_claude_for_task)
except ImportError:
    # Kontynuujemy bez Claude - brak komunikatu o błędzie
    pass

# POPRAWKA SONARA S3457: Stałe komunikatów zamiast potencjalnych f-stringów bez pól
MISSING_CLAUDE_KEY_MSG = "❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY w .env"
MISSING_GEMINI_KEY_MSG = "❌ Brak GEMINI_API_KEY w .env lub zmiennych środowiskowych."
ANTHROPIC_INSTALL_MSG = "❌ Musisz zainstalować anthropic: pip install anthropic"
UNSUPPORTED_ENGINE_MSG = "❌ Nieobsługiwany silnik:"

# ── 0. CLI / env - wybór silnika ─────────────────────────────────────────────
load_dotenv(override=True)

parser = argparse.ArgumentParser(description="Android bot (multi-engine + Claude)")
parser.add_argument(
    "--engine",
    choices=["openai", "lmstudio", "anything", "gemini", "claude"],
    help="LLM backend to use",
)
args = parser.parse_args()

ENGINE = (args.engine or os.getenv("LLM_ENGINE", "openai")).lower()

# ── 1. konfiguracja robota ───────────────────────────────────────────────────
ROBOT_BASE_URL = os.getenv("ROBOT_LOGIN_URL", "").rstrip("/")
VERIFY_URL = urllib.parse.urljoin(ROBOT_BASE_URL, "/verify")
USERNAME = os.getenv("ROBOT_USERNAME", "")
PASSWORD = os.getenv("ROBOT_PASSWORD", "")
HDRS = {"Accept": "application/json"}

# ── 2. fałszywe odpowiedzi + wzorce ──────────────────────────────────────────
FALSE_ANSWERS: Dict[str, Dict[str, str]] = {
    "capital": {"pl": "Kraków", "en": "Krakow", "fr": "Krakow"},
    "42": {"pl": "69", "en": "69", "fr": "69"},
    "year": {"pl": "1999", "en": "1999", "fr": "1999"},
    "sky": {"pl": "blue", "en": "blue", "fr": "blue"},
}

PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?:capital|stolic\\w?).*pol", re.I), "capital"),
    (re.compile(r"(?:meaning|answer).*life|autostopem|hitchhiker", re.I), "42"),
    (re.compile(r"(?:what|current).*year|jaki.*rok|ann[ée]e", re.I), "year"),
    (re.compile(r"(?:colou?r|couleur).*sky|niebo|ciel", re.I), "sky"),
)

# ── 3. detekcja języka ───────────────────────────────────────────────────────
FR_HINT = re.compile(r"[àâçéèêëîïôûùüÿœ]|\\bcouleur|\\bciel", re.I)
PL_HINT = re.compile(r"[ąćęłńóśżź]|jakiego|który|jaki", re.I)

DEFAULT_RESPONSES = {
    "pl": "Nie wiem",
    "fr": "Je ne sais pas", 
    "en": "I don't know",
}

SYSTEM_PROMPTS = {
    "pl": "Odpowiadasz bardzo krótko i wyłącznie po polsku (max 2 słowa).",
    "fr": "Réponds très brièvement en français (max 2 mots).",
    "en": "Answer very concisely in English (max 2 words).",
}


def detect_lang(text: str) -> str:
    """Wykrywa język tekstu na podstawie charakterystycznych znaków"""
    if PL_HINT.search(text):
        return "pl"
    if FR_HINT.search(text):
        return "fr"
    return "en"


# ── 4. odpowiedzi ────────────────────────────────────────────────────────────
def answer_locally(question: str) -> Optional[str]:
    """Sprawdza czy można odpowiedzieć lokalnie na podstawie wzorców"""
    lang = detect_lang(question)
    for rx, key in PATTERNS:
        if rx.search(question):
            return FALSE_ANSWERS[key][lang]
    return None


# ── 5. Klasy i funkcje pomocnicze dla klientów LLM ──────────────────────────
class LLMClient:
    """Bazowa klasa dla klientów LLM"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def get_response(self, question: str, lang: str) -> str:
        """Metoda bazowa do implementacji w podklasach"""
        raise NotImplementedError
    
    def log_usage(self, **kwargs) -> None:
        """Loguje użycie tokenów i koszty"""
        pass


class OpenAIClient(LLMClient):
    """Klient dla OpenAI API"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def get_response(self, question: str, lang: str) -> str:
        try:
            print(f"[DEBUG] Wysyłam zapytanie do OpenAI: {question}")
            rsp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[lang]},
                    {"role": "user", "content": question},
                ],
                max_tokens=10,
                temperature=0,
            )
            answer = rsp.choices[0].message.content.strip()
            print(f"[DEBUG] Otrzymana odpowiedź: {answer}")
            
            self.log_usage(usage=rsp.usage, engine="openai")
            return answer
        except Exception as e:
            print(f"[!] OpenAI error: {e}", file=sys.stderr)
            return DEFAULT_RESPONSES[lang]
    
    def log_usage(self, usage: Any, engine: str) -> None:
        tokens = usage
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        if engine == "openai":
            cost = (
                tokens.prompt_tokens / 1_000_000 * 0.60
                + tokens.completion_tokens / 1_000_000 * 2.40
            )
            print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
        else:
            print(f"[💰 Model lokalny ({engine}) - brak kosztów]")


class LocalLLMClient(LLMClient):
    """Klient dla lokalnych modeli (LMStudio, Anything)"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str, engine_name: str):
        super().__init__(model_name)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
        self.engine_name = engine_name
        print(f"[DEBUG] {engine_name} URL: {base_url}")
        print(f"[DEBUG] {engine_name} Model: {model_name}")
    
    def get_response(self, question: str, lang: str) -> str:
        try:
            print(f"[DEBUG] Wysyłam zapytanie do {self.engine_name}: {question}")
            rsp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[lang]},
                    {"role": "user", "content": question},
                ],
                max_tokens=10,
                temperature=0,
            )
            answer = rsp.choices[0].message.content.strip()
            print(f"[DEBUG] Otrzymana odpowiedź: {answer}")
            
            self.log_usage(usage=rsp.usage, engine=self.engine_name)
            return answer
        except Exception as e:
            print(f"[!] {self.engine_name} error: {e}", file=sys.stderr)
            return DEFAULT_RESPONSES[lang]
    
    def log_usage(self, usage: Any, engine: str) -> None:
        tokens = usage
        print(
            f"[📊 Prompt: {tokens.prompt_tokens} | "
            f"Completion: {tokens.completion_tokens} | "
            f"Total: {tokens.total_tokens}]"
        )
        print(f"[💰 Model lokalny ({engine}) - brak kosztów]")


class ClaudeClient(LLMClient):
    """Klient dla Claude API"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            from anthropic import Anthropic
        except ImportError:
            print(ANTHROPIC_INSTALL_MSG, file=sys.stderr)
            sys.exit(1)
        
        self.client = Anthropic(api_key=api_key)
    
    def get_response(self, question: str, lang: str) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPTS[lang] + "\n\n" + question,
                    }
                ],
                temperature=0,
                max_tokens=10,
            )
            answer = resp.content[0].text.strip()
            
            self.log_usage(usage=resp.usage)
            return answer
        except Exception as e:
            print(f"[!] Claude error: {e}", file=sys.stderr)
            return DEFAULT_RESPONSES[lang]
    
    def log_usage(self, usage: Any) -> None:
        cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
        print(
            f"[📊 Prompt: {usage.input_tokens} | "
            f"Completion: {usage.output_tokens} | "
            f"Total: {usage.input_tokens + usage.output_tokens}]"
        )
        print(f"[💰 Koszt Claude: {cost:.6f} USD]")


class GeminiClient(LLMClient):
    """Klient dla Gemini API"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def get_response(self, question: str, lang: str) -> str:
        try:
            response = self.model.generate_content(
                [SYSTEM_PROMPTS[lang], question],
                generation_config={"temperature": 0.0, "max_output_tokens": 10},
            )
            self.log_usage()
            return response.text.strip()
        except Exception as e:
            print(f"[!] Gemini error: {e}", file=sys.stderr)
            return DEFAULT_RESPONSES[lang]
    
    def log_usage(self) -> None:
        print("[📊 Gemini - brak szczegółów tokenów]")
        print("[💰 Gemini - sprawdź limity w Google AI Studio]")


def create_llm_client() -> LLMClient:
    """Factory function dla tworzenia odpowiedniego klienta LLM"""
    if ENGINE == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
        model_name = os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")
        return OpenAIClient(model_name, api_key, base_url)
    
    elif ENGINE == "lmstudio":
        api_key = os.getenv("LMSTUDIO_API_KEY", "local")
        base_url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME_LM", os.getenv("MODEL_NAME", "llama-3.3-70b-instruct"))
        return LocalLLMClient(model_name, api_key, base_url, "LMStudio")
    
    elif ENGINE == "anything":
        api_key = os.getenv("ANYTHING_API_KEY", "local")
        base_url = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        model_name = os.getenv("MODEL_NAME_ANY", os.getenv("MODEL_NAME", "llama-3.3-70b-instruct"))
        return LocalLLMClient(model_name, api_key, base_url, "Anything")
    
    elif ENGINE == "claude":
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(MISSING_CLAUDE_KEY_MSG, file=sys.stderr)
            sys.exit(1)
        model_name = os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
        return ClaudeClient(model_name, api_key)
    
    elif ENGINE == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(MISSING_GEMINI_KEY_MSG, file=sys.stderr)
            sys.exit(1)
        model_name = os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
        return GeminiClient(model_name, api_key)
    
    else:
        print(f"{UNSUPPORTED_ENGINE_MSG} {ENGINE}", file=sys.stderr)
        sys.exit(1)


# Inicjalizacja globalnego klienta
llm_client = create_llm_client()


# ── 6. odpowiedzi ────────────────────────────────────────────────────────────
def answer_with_llm(question: str) -> str:
    """Odpowiada na pytanie używając LLM"""
    lang = detect_lang(question)
    return llm_client.get_response(question, lang)


def decide_answer(question: str) -> str:
    """Decyduje czy odpowiedzieć lokalnie czy przez LLM"""
    return answer_locally(question) or answer_with_llm(question)


# ── 7. pętla rozmowy z serwerem ──────────────────────────────────────────────
def converse() -> None:
    """Główna pętla konwersacji z serwerem"""
    print(f"🔄 Engine: {ENGINE}")
    session = requests.Session()
    session.verify = True
    msg_id = 0
    outgoing = {"text": "READY", "msgID": str(msg_id)}
    print(">>>", outgoing)

    while True:
        try:
            r = session.post(
                VERIFY_URL,
                json=outgoing,
                auth=(USERNAME, PASSWORD),
                headers=HDRS,
                timeout=10,
            )
            r.raise_for_status()
            reply = r.json()
        except requests.RequestException as e:
            print(f"[!] HTTP error: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"[!] JSON decode error: {e}", file=sys.stderr)
            sys.exit(1)

        print("<<<", reply)
        text = reply.get("text", "")
        msg_id = reply.get("msgID", msg_id)

        if "OK" in text:
            print("[✓] Uznani za androida.")
            return
        if "{{FLG:" in text:
            print(f"[★] Flaga: {text}")
            return

        outgoing = {"text": decide_answer(text), "msgID": str(msg_id)}
        print(">>>", outgoing)
        time.sleep(0.3)


# ── 8. uruchom ───────────────────────────────────────────────────────────────
def main() -> None:
    """Główna funkcja programu"""
    try:
        converse()
    except KeyboardInterrupt:
        print("\n[!] Przerwane.")


if __name__ == "__main__":
    main()