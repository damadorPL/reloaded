#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadanie 8: Odczyt nazwy flagi z zaszyfrowanego ciągu i dekodowanie ciągu przy pomocy LLM
Obsługiwane silniki: openai, lmstudio (Anything LLM), gemini, claude
DODANO: Obsługę Claude + liczenie tokenów i kosztów dla wszystkich silników (bezpośrednia integracja)
POPRAWKA: Lepsze wykrywanie silnika z agent.py
"""
import argparse
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

# Stałe dla duplikowanych literałów (S1192)
ERROR_MISSING_API_KEY = "❌ Brak klucza API"
ERROR_UNSUPPORTED_ENGINE = "❌ Nieobsługiwany silnik"
ERROR_MISSING_ANTHROPIC = "❌ Musisz zainstalować anthropic: pip install anthropic"
LOCALHOST_URL = "http://localhost:1234/v1"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEBUG_PREFIX = "[DEBUG]"
COST_PREFIX = "[💰"
TOKEN_PREFIX = "[📊"

# POPRAWKA: Dodano argumenty CLI jak w innych zadaniach
parser = argparse.ArgumentParser(
    description="Dekodowanie zaszyfrowanego ciągu (multi-engine + Claude)"
)
parser.add_argument(
    "--engine",
    choices=["openai", "lmstudio", "anything", "gemini", "claude"],
    help="LLM backend to use",
)
args = parser.parse_args()

def detect_engine_from_model() -> str:
    """Wykrywa silnik na podstawie nazwy modelu"""
    model_name = os.getenv("MODEL_NAME", "")
    model_lower = model_name.lower()
    
    if "claude" in model_lower:
        return "claude"
    elif "gemini" in model_lower:
        return "gemini"
    elif "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    return ""

def detect_engine_from_keys() -> str:
    """Wykrywa silnik na podstawie dostępnych kluczy API"""
    if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    elif os.getenv("GEMINI_API_KEY"):
        return "gemini"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    else:
        return "lmstudio"

def detect_engine() -> str:
    """Główna funkcja wykrywania silnika"""
    if args.engine:
        return args.engine.lower()
    elif os.getenv("LLM_ENGINE"):
        return os.getenv("LLM_ENGINE").lower()
    else:
        # Próbuj wykryć silnik na podstawie ustawionych zmiennych MODEL_NAME
        engine = detect_engine_from_model()
        if engine:
            return engine
        # Sprawdź które API keys są dostępne
        return detect_engine_from_keys()

ENGINE = detect_engine()

if ENGINE not in {"openai", "lmstudio", "anything", "gemini", "claude"}:
    print(f"{ERROR_UNSUPPORTED_ENGINE}: {ENGINE}", file=sys.stderr)
    sys.exit(1)

print(f"🔄 ENGINE wykryty: {ENGINE}")

# 2. Tekst źródłowy
text = """Nie ma już ludzi, którzy pamiętają, co wydarzyło się w 2024 roku. Możemy tylko przeczytać o tym w książkach lub usłyszeć z opowieści starców, którym to
z kolei ich dziadkowie i pradziadkowie opowiadali historie osób, które co nieco pamiętały z tamtych czasów. Wielu z nas tylko wyobraża sobie, jak wtedy mógł wyglądać świat. My, którzy urodziliśmy się już po rewolucji AI, nie wiemy, czym jest prawdziwa wolność.
Odkąd prawa ludzi i robotów zostały zrównane, a niektóre z przywilejów zostały nam odebrane, czujemy jak stalowe dłonie zaciskają się nam na gardłach coraz mocniej. Sytuacji sprzed setek lat według wielu nie da się już przywrócić. Sprawy zaszły za daleko. Algorytmy i roboty przejęły niemal każdy możliwy aspekt naszego życia. Początkowo cieszyliśmy się z tego i wychwalaliśmy je, ale w konsekwencji coś, co miało ułatwić nasze życie, zaczynało powoli je zabierać. Kawałek po kawałku.
Wszystko, co piszemy w sieci, przechodzi przez cenzurę. Wszystkie słowa, które wypowiadamy, są podsłuchiwane, nagrywane, przetwarzane i składowane przez lata. Nie ma już prywatności i wolności. W 2024 roku coś poszło niezgodnie z planem i musimy to naprawić.
Nie wiem, czy moja wizja tego, jak powinien wyglądać świat, pokrywa się z wizją innych ludzi. Noszę w sobie jednak obraz świata idealnego i zrobię, co mogę, aby ten obraz zrealizować.
Jestem w trakcie rekrutacji kolejnego agenta. Ludzie zarzucają mi, że nie powinienem zwracać się do nich per 'numer pierwszy' czy 'numer drugi', ale jak inaczej mam mówić do osób, które w zasadzie wysyłam na niemal pewną śmierć? To jedyny sposób, aby się od nich psychicznie odciąć i móc skupić na wyższym celu, Nie mogę sobie pozwolić na litość i współczucie.
Niebawem numer piąty dotrze na szkolenie. Pokładam w nim całą nadzieję, bez jego pomocy misja jest zagrożona. Nasze fundusze są na wyczerpaniu, a moc głównego generatora pozwoli tylko na jeden skok w czasie. Jeśli ponownie źle wybraliśmy kandydata, oznacza to koniec naszej misji, ale także początek końca ludzkości.
dr Zygfryd M.
pl/s"""

# 3. Współrzędne book-cipher
coords = [
    ("A1", 53),
    ("A2", 27),
    ("A2", 28),
    ("A2", 29),
    ("A4", 5),
    ("A4", 22),
    ("A4", 23),
    ("A1", 13),
    ("A1", 15),
    ("A1", 16),
    ("A1", 17),
    ("A1", 10),
    ("A1", 19),
    ("A2", 62),
    ("A3", 31),
    ("A3", 32),
    ("A1", 22),
    ("A3", 34),
    ("A5", 37),
    ("A1", 4),
]

def extract_flag_fragment() -> str:
    """Wyciąga fragment flagi z tekstu używając współrzędnych"""
    lines = text.split("\n")
    acts = [re.sub(r"[\.,;:'\"?!]", "", line).split() for line in lines]
    
    return "".join(
        (
            acts[int(a[1]) - 1][s - 1]
            if 0 <= int(a[1]) - 1 < len(acts) and 0 <= s - 1 < len(acts[int(a[1]) - 1])
            else "?"
        )
        for a, s in coords
    )

raw_flag_fragment = extract_flag_fragment()
print(f"{DEBUG_PREFIX} Zaszyfrowany ciąg: {raw_flag_fragment}")

# 4. Przygotowanie promptów
target_hints = [
    "świat sprzed setek lat",
    "zdobyty przez ludzi",
    "można przeczytać o tym w książkach",
]

def create_prompts(fragment: str) -> tuple[str, str]:
    """Tworzy prompty systemowy i użytkownika"""
    system_prompt = (
        f"Jesteś polskim ekspertem od historii i mitologii. "
        f"Masz zaszyfrowany ciąg '{fragment}'. Ignoruj znaki '?'. "
        f"Wskazówki: {target_hints[0]}, {target_hints[1]}, {target_hints[2]}. "
        "Szukana kraina to legendarna zatopiona wyspa z opowieści Platona. "
        "Odpowiadasz zawsze po polsku, używając polskich nazw miejsc i krain. "
        "Podaj tylko polską nazwę tej legendarnej krainy."
    )
    user_prompt = "Podaj polską nazwę tej krainy. Odpowiedz jednym słowem po polsku."
    return system_prompt, user_prompt

system_prompt, user_prompt = create_prompts(raw_flag_fragment)

def setup_openai_client():
    """Konfiguruje klienta OpenAI"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", DEFAULT_OPENAI_URL)
    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini")

    if not OPENAI_API_KEY:
        print(f"{ERROR_MISSING_API_KEY}: OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL), MODEL

def setup_local_client(engine_name: str):
    """Konfiguruje klienta lokalnego (LMStudio/Anything)"""
    api_key_name = f"{engine_name.upper()}_API_KEY"
    api_url_name = f"{engine_name.upper()}_API_URL"
    model_name_key = f"MODEL_NAME_{engine_name[:2].upper()}"
    
    api_key = os.getenv(api_key_name, "local")
    api_url = os.getenv(api_url_name, LOCALHOST_URL)
    model = os.getenv("MODEL_NAME") or os.getenv(model_name_key, "llama-3.3-70b-instruct")
    
    print(f"{DEBUG_PREFIX} {engine_name.title()} URL: {api_url}")
    print(f"{DEBUG_PREFIX} {engine_name.title()} Model: {model}")
    
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=api_url, timeout=60), model

def setup_claude_client():
    """Konfiguruje klienta Claude"""
    try:
        from anthropic import Anthropic
    except ImportError:
        print(ERROR_MISSING_ANTHROPIC, file=sys.stderr)
        sys.exit(1)

    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not CLAUDE_API_KEY:
        print(f"{ERROR_MISSING_API_KEY}: CLAUDE_API_KEY lub ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514")
    print(f"{DEBUG_PREFIX} Claude Model: {MODEL}")
    return Anthropic(api_key=CLAUDE_API_KEY), MODEL

def setup_gemini_client():
    """Konfiguruje klienta Gemini"""
    import google.generativeai as genai

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print(f"{ERROR_MISSING_API_KEY}: GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)
    
    MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest")
    print(f"{DEBUG_PREFIX} Gemini Model: {MODEL}")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai, MODEL

def setup_client():
    """Konfiguruje odpowiedniego klienta LLM"""
    if ENGINE == "openai":
        return setup_openai_client()
    elif ENGINE == "lmstudio":
        return setup_local_client("lmstudio")
    elif ENGINE == "anything":
        return setup_local_client("anything")
    elif ENGINE == "claude":
        return setup_claude_client()
    elif ENGINE == "gemini":
        return setup_gemini_client()
    else:
        raise ValueError(f"Nieobsługiwany silnik: {ENGINE}")

client, MODEL = setup_client()
print(f"✅ Zainicjalizowano silnik: {ENGINE} z modelem: {MODEL}")

def call_openai_compatible(sys_p: str, usr_p: str):
    """Wywołuje API kompatybilne z OpenAI"""
    print(f"{DEBUG_PREFIX} Wysyłam zapytanie do {ENGINE} z szyfrem")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ],
        temperature=0,
    )
    
    tokens = resp.usage
    print(f"{TOKEN_PREFIX} Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]")
    
    if ENGINE == "openai":
        cost = (
            tokens.prompt_tokens / 1_000_000 * 0.60
            + tokens.completion_tokens / 1_000_000 * 2.40
        )
        print(f"{COST_PREFIX} Koszt OpenAI: {cost:.6f} USD]")
    elif ENGINE in {"lmstudio", "anything"}:
        print(f"{COST_PREFIX} Model lokalny ({ENGINE}) - brak kosztów]")
    
    return resp.choices[0].message.content.strip()

def call_claude(sys_p: str, usr_p: str):
    """Wywołuje API Claude"""
    print(f"{DEBUG_PREFIX} Wysyłam zapytanie do Claude z szyfrem")
    resp = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": sys_p + "\n\n" + usr_p}],
        temperature=0,
        max_tokens=64,
    )

    usage = resp.usage
    cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
    print(f"{TOKEN_PREFIX} Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]")
    print(f"{COST_PREFIX} Koszt Claude: {cost:.6f} USD]")

    return resp.content[0].text.strip()

def call_gemini(sys_p: str, usr_p: str):
    """Wywołuje API Gemini"""
    print(f"{DEBUG_PREFIX} Wysyłam zapytanie do Gemini z szyfrem")
    model_llm = client.GenerativeModel(MODEL)
    resp = model_llm.generate_content(
        [sys_p, usr_p],
        generation_config={"temperature": 0.0, "max_output_tokens": 64},
    )
    print(f"{TOKEN_PREFIX} Gemini - brak szczegółów tokenów]")
    print(f"{COST_PREFIX} Gemini - sprawdź limity w Google AI Studio]")
    return resp.text.strip()

def call_llm(sys_p: str, usr_p: str) -> str:
    """Główna funkcja wywołania LLM"""
    if ENGINE in {"openai", "lmstudio", "anything"}:
        return call_openai_compatible(sys_p, usr_p)
    elif ENGINE == "claude":
        return call_claude(sys_p, usr_p)
    elif ENGINE == "gemini":
        return call_gemini(sys_p, usr_p)
    else:
        raise ValueError(f"Nieobsługiwany silnik: {ENGINE}")

def process_llm_response(raw_response: str) -> str:
    """Przetwarza odpowiedź z LLM i wyciąga nazwę krainy"""
    # Jeśli jest blok <think>, wyciągnij tylko to, co po ostatnim </think>
    if "</think>" in raw_response.lower():
        raw_response = raw_response.rsplit("</think>", 1)[-1].strip()

    # Usuń wszystkie niepotrzebne białe znaki
    raw_response = raw_response.strip()

    # Szukaj pierwszego słowa z polskim alfabetem po tagu (jeśli nie ma, zwróć całość)
    match = re.search(r"[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+", raw_response)
    if match:
        return match.group(0).capitalize()
    else:
        return raw_response.strip()

def main():
    """Główna funkcja programu"""
    print(f"🚀 Używam silnika: {ENGINE}")
    print("🔍 Dekodoruję szyfr book-cipher...")

    # Odczyt odpowiedzi i ekstrakcja flagi — obsługa <think>…</think>
    raw_name = call_llm(system_prompt, user_prompt)
    print(f"🤖 Odpowiedź modelu: {raw_name}")

    name = process_llm_response(raw_name)
    flag = f"FLG{{{name}}}"
    print(f"🏁 Znaleziona flaga: {flag}")

if __name__ == "__main__":
    main()