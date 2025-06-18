#!/usr/bin/env python3
"""
S02E04 - Klasyfikacja plików z fabryki: jedna prośba do LLM na plik, detekcja języka, zwracanie kategorii
• Multiengine: openai / gemini / lmstudio / anything / claude
• Ekstrakcja: txt→tekst, mp3/wav→Whisper lokalnie, png/jpg→OCR (OpenCV+pytesseract)
• Orkiestracja: LangGraph

POPRAWKI: Konserwatywna logika klasyfikacji people - tylko potwierdzone schwytania
POPRAWKA: Lepsze wykrywanie silnika z agent.py
POPRAWKA SONARA: Refaktoryzacja funkcji wysokiej złożoności kognitywnej, stałe dla duplikatów, obsługa wyjątków
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import pytesseract
import requests
import whisper
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from langgraph.graph import END, START, StateGraph

# POPRAWKA SONARA S1192: Stałe dla duplikowanych literałów
FOUND_ONE_GUY = "found one guy"
ANTHROPIC_INSTALL_MSG = "❌ Musisz zainstalować anthropic: pip install anthropic"
CAPTURED_KEYWORD = "captured"
TRANSMITTER_KEYWORD = "nadajnik"
FINGERPRINT_KEYWORD = "odcisk"

# Słowa kluczowe dla heurystyk
PRESENCE_KEYWORDS = [
    FOUND_ONE_GUY,
    CAPTURED_KEYWORD,
    "schwytanych",
    "wykryto jednostkę organiczną",
    "przedstawił się jako",
    "infiltrator",
    "organiczna",
    "ultradźwięk",
    "osobnik",
    "przechwyc",
]

HARDWARE_KEYWORDS = [
    "napraw",
    "uster",
    "naprawa anteny",
    "wymiana ogniw",
    "usterka spowodowana",
    "zwarcie kabli",
    "uszkodzenie",
]

SOFTWARE_KEYWORDS = [
    "aktualizacja systemu",
    "system update",
    "software",
    "algorytm",
    "moduł ai",
]

# --- 1. Konfiguracja i inicjalizacja LLM ---
load_dotenv(override=True)

# POPRAWKA: Dodano argumenty CLI jak w innych zadaniach
parser = argparse.ArgumentParser(
    description="Klasyfikacja plików z fabryki (multi-engine + Claude)"
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


def get_model_name(engine: str) -> str:
    """Zwraca nazwę modelu dla danego silnika"""
    model_mappings = {
        "openai": os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_OPENAI", "gpt-4o-mini"),
        "claude": os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514"),
        "gemini": os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest"),
        "lmstudio": os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_LM", "llama-3.3-70b-instruct"),
        "anything": os.getenv("MODEL_NAME") or os.getenv("MODEL_NAME_ANY", "llama-3.3-70b-instruct"),
    }
    return model_mappings.get(engine, "")


def validate_api_keys(engine: str) -> None:
    """Sprawdza czy wymagane API keys są dostępne"""
    if engine == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ Brak OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)
    elif engine == "claude" and not (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ Brak CLAUDE_API_KEY lub ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)
    elif engine == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print("❌ Brak GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)


# Inicjalizacja silnika
ENGINE = detect_engine()
validate_engine(ENGINE)
print(f"🔄 ENGINE wykryty: {ENGINE}")

CENTRALA_API_KEY = os.getenv("CENTRALA_API_KEY")
FABRYKA_URL = os.getenv("FABRYKA_URL")
REPORT_URL = os.getenv("REPORT_URL")

if not all([CENTRALA_API_KEY, FABRYKA_URL, REPORT_URL]):
    print(
        "❌ Brak wymaganych zmiennych: CENTRALA_API_KEY, FABRYKA_URL, REPORT_URL",
        file=sys.stderr,
    )
    sys.exit(1)

MODEL_NAME = get_model_name(ENGINE)
if not MODEL_NAME:
    print(f"❌ Brak MODEL_NAME dla silnika {ENGINE}", file=sys.stderr)
    sys.exit(1)

validate_api_keys(ENGINE)

# klucze i URL-e
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "local")
LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
ANYTHING_API_KEY = os.getenv("ANYTHING_API_KEY", "local")
ANYTHING_API_URL = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")

# inicjalizacja klienta
if ENGINE == "openai":
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1"),
    )

elif ENGINE == "claude":
    # Bezpośrednia integracja Claude
    try:
        from anthropic import Anthropic
    except ImportError:
        print(ANTHROPIC_INSTALL_MSG, file=sys.stderr)
        sys.exit(1)

    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    claude_client = Anthropic(api_key=CLAUDE_API_KEY)

elif ENGINE == "gemini":
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print(f"✅ Zainicjalizowano silnik: {ENGINE} z modelem: {MODEL_NAME}")

# --- 2. Init Whisper ---
audio_model = whisper.load_model(os.getenv("WHISPER_MODEL", "small"))


# --- 3. Funkcje pomocnicze LLM ---
def has_classification_keywords(text: str) -> bool:
    """Sprawdza czy tekst zawiera słowa kluczowe klasyfikacji"""
    import re
    keywords = re.findall(r"\b(people|hardware|other)\b", text.lower())
    return bool(keywords)


def extract_classification_keyword(text: str) -> Optional[str]:
    """Wyciąga pierwsze słowo kluczowe klasyfikacji z tekstu"""
    import re
    keywords = re.findall(r"\b(people|hardware|other)\b", text.lower())
    return keywords[0] if keywords else None


def call_llm_openai(prompt: str) -> str:
    """Wywołuje OpenAI API"""
    print("[DEBUG] Wysyłam zapytanie do OpenAI")
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    # Liczenie tokenów
    tokens = resp.usage
    print(
        f"[📊 Prompt: {tokens.prompt_tokens} | Completion: {tokens.completion_tokens} | Total: {tokens.total_tokens}]"
    )
    cost = (
        tokens.prompt_tokens / 1_000_000 * 0.60
        + tokens.completion_tokens / 1_000_000 * 2.40
    )
    print(f"[💰 Koszt OpenAI: {cost:.6f} USD]")
    return resp.choices[0].message.content.strip().lower()


def call_llm_claude(prompt: str) -> str:
    """Wywołuje Claude API"""
    print("[DEBUG] Wysyłam zapytanie do Claude")
    resp = claude_client.messages.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=32,  # Krótka odpowiedź dla klasyfikacji
    )

    # Liczenie tokenów Claude
    usage = resp.usage
    cost = usage.input_tokens * 0.00003 + usage.output_tokens * 0.00015
    print(
        f"[📊 Prompt: {usage.input_tokens} | Completion: {usage.output_tokens} | Total: {usage.input_tokens + usage.output_tokens}]"
    )
    print(f"[💰 Koszt Claude: {cost:.6f} USD]")

    return resp.content[0].text.strip().lower()


def call_llm_gemini(prompt: str) -> str:
    """Wywołuje Gemini API"""
    print("[DEBUG] Wysyłam zapytanie do Gemini")
    response = genai.GenerativeModel(MODEL_NAME).generate_content(
        prompt, generation_config={"temperature": 0.0, "max_output_tokens": 32}
    )
    print("[📊 Gemini - brak szczegółów tokenów]")
    print("[💰 Gemini - sprawdź limity w Google AI Studio]")
    return response.text.strip().lower()


def call_llm_lmstudio(prompt: str) -> str:
    """Wywołuje LMStudio API"""
    print("[DEBUG] Wysyłam zapytanie do LMStudio")
    url = LMSTUDIO_API_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5,
        "stop": ["\n", ".", " "],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("[📊 LMStudio - brak szczegółów tokenów]")
    print("[💰 LMStudio - model lokalny, brak kosztów]")
    return content.strip().lower()


def call_llm_anything(prompt: str) -> str:
    """Wywołuje Anything API"""
    print("[DEBUG] Wysyłam zapytanie do Anything")
    headers = {
        "Authorization": f"Bearer {ANYTHING_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL_NAME, "inputs": prompt}
    resp = requests.post(ANYTHING_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    print("[📊 Anything - brak szczegółów tokenów]")
    print("[💰 Anything - model lokalny, brak kosztów]")
    return resp.json().get("generated_text", "").strip().lower()


def call_llm(prompt: str) -> str:
    """Wywołuje odpowiedni LLM na podstawie ENGINE"""
    if ENGINE == "openai":
        return call_llm_openai(prompt)
    elif ENGINE == "claude":
        return call_llm_claude(prompt)
    elif ENGINE == "gemini":
        return call_llm_gemini(prompt)
    elif ENGINE == "lmstudio":
        return call_llm_lmstudio(prompt)
    elif ENGINE == "anything":
        return call_llm_anything(prompt)
    else:
        raise ValueError(f"Nieobsługiwany silnik: {ENGINE}")


def call_llm_with_retry(prompt: str, max_retries: int = 2) -> str:
    """
    POPRAWKA SONARA S3776: Wywołuje LLM z retry logic - refaktoryzacja funkcji o wysokiej złożoności
    """
    last_result = ""
    
    for attempt in range(max_retries + 1):
        try:
            result = call_llm(prompt)
            last_result = result
            
            # Sprawdź czy odpowiedź zawiera oczekiwane słowa kluczowe
            if has_classification_keywords(result):
                return result
                
            # Jeśli nie znaleziono słów kluczowych i to nie ostatnia próba    
            if attempt < max_retries:
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} - no valid keywords found, retrying...")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} - Request failed: {e}")
            if attempt < max_retries:
                print("Retrying...")
                continue
            else:
                print("[ERROR] All retry attempts failed due to request errors")
                
        except (ValueError, KeyError, AttributeError) as e:
            print(f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} - API response error: {e}")
            if attempt < max_retries:
                print("Retrying...")
                continue
            else:
                print("[ERROR] All retry attempts failed due to API errors")
                
        except Exception as e:
            print(f"[RETRY] Attempt {attempt + 1}/{max_retries + 1} - Unexpected error: {e}")
            if attempt < max_retries:
                print("Retrying...")
                continue
            else:
                print("[ERROR] All retry attempts failed due to unexpected errors")

    return last_result  # Zwróć ostatnią odpowiedź nawet jeśli błędną


# --- 4. Ekstrakcja zawartości ---
def download_and_extract(dest: Path) -> None:
    if (dest / "2024-11-12_report-00-sektor_C4.txt").exists():
        print("[INFO] Pliki już rozpakowane - pomijam pobieranie.")
        return
    dest.mkdir(exist_ok=True)
    zip_path = dest / "fabryka.zip"
    print("[INFO] Pobieram dane z fabryki…")
    resp = requests.get(FABRYKA_URL, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()


def extract_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")


def extract_audio(fp: Path) -> str:
    result = audio_model.transcribe(str(fp))
    text = result.get("text", "")
    Path("debug").mkdir(exist_ok=True)
    with open(f"debug/{fp.name}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text


def extract_image(fp: Path) -> str:
    """POPRAWKA SONARA S5754: Określenie konkretnych typów wyjątków"""
    try:
        img = cv2.imread(str(fp))
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, lang="pol")
    except (cv2.error, pytesseract.TesseractError, OSError, IOError) as e:
        print(f"[WARNING] Error extracting image {fp}: {e}")
        return ""


# --- 5. Detekcja języka ---
def detect_language(text: str) -> str:
    """POPRAWKA SONARA S5754: Określenie konkretnego typu wyjątku"""
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "en"
    return "pl" if lang.startswith("pl") else "en"


# --- 6. Funkcje heurystyczne klasyfikacji ---
def check_people_heuristics(text_lower: str) -> bool:
    """Sprawdza heurystyki dla kategorii 'people'"""
    # Nadajnik + odciski palców = definitywnie people
    if (TRANSMITTER_KEYWORD in text_lower or "transmitter" in text_lower) and (
        FINGERPRINT_KEYWORD in text_lower or "fingerprint" in text_lower
    ):
        print("[HEURISTIC] People detected: transmitter with fingerprints")
        return True
    
    # Podstawowe słowa dla people
    found_keywords = [kw for kw in PRESENCE_KEYWORDS if kw in text_lower]
    if found_keywords:
        print(f"[HEURISTIC] People detected: {found_keywords}")
        return True
    
    return False


def check_hardware_heuristics(text_lower: str) -> bool:
    """Sprawdza heurystyki dla kategorii 'hardware'"""
    found_keywords = [kw for kw in HARDWARE_KEYWORDS if kw in text_lower]
    if found_keywords:
        print(f"[HEURISTIC] Hardware repair detected: {found_keywords}")
        return True
    return False


def check_software_heuristics(text_lower: str) -> bool:
    """Sprawdza heurystyki dla kategorii 'software' (klasyfikowane jako 'other')"""
    found_keywords = [kw for kw in SOFTWARE_KEYWORDS if kw in text_lower]
    if found_keywords:
        print(f"[HEURISTIC] Software update detected (not hardware): {found_keywords}")
        return True
    return False


def apply_engine_heuristics(text_lower: str, engine: str) -> Optional[str]:
    """Aplikuje heurystyki specyficzne dla danego silnika"""
    if engine == "openai":
        if check_people_heuristics(text_lower):
            return "people"
    elif engine == "lmstudio":
        if check_hardware_heuristics(text_lower):
            return "hardware"
        if check_people_heuristics(text_lower):
            return "people"
    elif engine == "gemini":
        if check_people_heuristics(text_lower):
            return "people"
        if check_hardware_heuristics(text_lower):
            return "hardware"
    elif engine == "claude":
        if check_people_heuristics(text_lower):
            return "people"
        if check_software_heuristics(text_lower):
            return "other"
        if check_hardware_heuristics(text_lower):
            return "hardware"
    
    return None


def create_classification_prompt(text: str, filename: str, lang: str) -> str:
    """Tworzy prompt dla klasyfikacji w zależności od silnika i języka"""
    if ENGINE in ("openai", "gemini", "claude"):
        if lang == "pl":
            return f"""
Plik: {filename}
Zawartość:
{text}

Zadanie: przypisz do jednej z kategorii:
- people (informacje o schwytanych ludziach lub ich śladach obecności)
- hardware (naprawione usterki hardware'u)
- other (wszystko inne)

Odpowiedz tylko: people/hardware/other.
Upewnij się, że klasyfikujesz tylko wtedy, gdy są wyraźne informacje o schwytanych osobach.
Jeśli to tylko poszukiwania lub brak wyników - zaklasyfikuj jako 'other'.
"""
        else:
            return f"""
File: {filename}
Content:
{text}

Task: classify into one category:
- people (notes about captured people or traces of their presence)
- hardware (fixed hardware malfunctions)
- other (anything else)

Answer only: people/hardware/other.
Only classify as 'people' if actual capture or presence is confirmed. 
Mere searches or absence should be classified as 'other'.
"""
    else:  # LMStudio/Anything
        if lang == "pl":
            return f"""
KLASYFIKACJA PLIKU:

Plik: {filename}
Treść: {text}

ZADANIE: Odpowiedz TYLKO jednym słowem: people, hardware lub other

ZASADY:
- people = schwytane osoby, wykryte obecności ludzi
- hardware = naprawy sprzętu, usterki, wymiany części  
- other = wszystko inne

ODPOWIEDŹ (tylko jedno słowo):"""
        else:
            return f"""
FILE CLASSIFICATION:

File: {filename}
Content: {text}

TASK: Answer with ONLY one word: people, hardware or other

RULES:
- people = captured persons, detected human presence
- hardware = equipment repairs, malfunctions, part replacements
- other = everything else

ANSWER (one word only):"""


def classify_file(text: str, filename: str) -> str:
    """
    POPRAWKA SONARA S3776: Klasyfikuje plik - refaktoryzacja funkcji o wysokiej złożoności
    """
    lang = detect_language(text)
    text_lower = text.lower()

    # Debug - pokaż fragment tekstu dla analizy
    print(f"[DEBUG] Text fragment: {text_lower[:200]}...")

    # Sprawdź heurystyki specyficzne dla silnika
    heuristic_result = apply_engine_heuristics(text_lower, ENGINE)
    if heuristic_result:
        return heuristic_result

    # Jeśli heurystyki nie dały wyniku, użyj LLM
    prompt = create_classification_prompt(text, filename, lang)
    result = call_llm_with_retry(prompt)
    
    # Post-processing odpowiedzi - wyciągnij tylko słowo kluczowe
    keyword = extract_classification_keyword(result)
    if keyword:
        print(f"[POST-PROCESS] Extracted keyword: {keyword}")
        return keyword
    else:
        print(f"[WARNING] Nieoczekiwana odpowiedź LLM: '{result[:100]}...' -> defaulting to 'other'")
        return "other"


# --- 7. Pipeline LangGraph ---
def download_node(state):
    """Node funkcja dla pobierania danych"""
    download_and_extract(Path("fabryka"))
    return state


def classify_node(state):
    """Node funkcja dla klasyfikacji plików"""
    root = Path("fabryka")
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and "facts" not in p.parts and p.name != "weapons_tests.zip"
    ]
    print(f"[CLASSIFY] Found {len(files)} files")
    cats = {"people": [], "hardware": [], "other": []}

    for fp in sorted(files):
        print(f"\n[CLASSIFY] Processing: {fp.name}")

        # Ekstrakcja tekstu
        if fp.suffix == ".txt":
            text = extract_text(fp)
        elif fp.suffix in [".mp3", ".wav"]:
            text = extract_audio(fp)
        elif fp.suffix in [".png", ".jpg", ".jpeg"]:
            text = extract_image(fp)
        else:
            text = ""

        # Debug snippet
        snippet = text.replace("\n", " ")[:100]
        print(f"[CLASSIFY] Snippet: {snippet!r}")

        # Klasyfikacja
        cat = classify_file(text, fp.name)
        print(f"[CLASSIFY] Result: {cat}")

        cats[cat].append(fp.name)

    # Zapis surowej klasyfikacji do debugowania
    Path("raw_classification.json").write_text(
        json.dumps(cats, ensure_ascii=False, indent=4), encoding="utf-8"
    )

    # Zwracamy stan z wynikami klasyfikacji
    state.update(cats)
    return state


def aggregate_node(state):
    """Node funkcja dla agregacji wyników"""
    ppl = sorted(state.get("people", []))
    hw = sorted(state.get("hardware", []))
    out = {"people": ppl, "hardware": hw}

    print(f"\n[AGGREGATE] Final results:")
    print(f"  People: {len(ppl)} files: {ppl}")
    print(f"  Hardware: {len(hw)} files: {hw}")
    print(f"  Other: {len(state.get('other', []))} files (not included in report)")

    Path("wynik.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    state["report"] = out
    return state


def send_node(state):
    """Node funkcja dla wysyłania raportu"""
    payload = {
        "task": "kategorie",
        "apikey": CENTRALA_API_KEY,
        "answer": state.get("report"),
    }
    Path("payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8"
    )

    print(f"\n[SEND] Sending payload: {payload}")
    resp = requests.post(REPORT_URL, json=payload)
    print(f"[SEND] Centralna odpowiedź: {resp.text}")
    return state


def build_graph():
    """Buduje graf LangGraph z właściwymi funkcjami"""
    graph = StateGraph(input=dict, output=dict)

    # Dodawanie node'ów
    graph.add_node("download", download_node)
    graph.add_edge(START, "download")

    graph.add_node("classify", classify_node)
    graph.add_edge("download", "classify")

    graph.add_node("aggregate", aggregate_node)
    graph.add_edge("classify", "aggregate")

    graph.add_node("send", send_node)
    graph.add_edge("aggregate", "send")
    graph.add_edge("send", END)

    return graph.compile()


# --- 8. main ---
def main():
    print("=== Zadanie 9: klasyfikacja plików z fabryki ===")
    print(f"🚀 Używam silnika: {ENGINE}")
    print("LOGIKA: Oryginalna + fix dla OpenAI (nadajnik z odciskami = people)")
    print("OCZEKIWANE: people=3, hardware=3 pliki")
    print("Startuje pipeline...\n")
    build_graph().invoke({})


if __name__ == "__main__":
    main()