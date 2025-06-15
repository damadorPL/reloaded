## Uwaga - easteregg działa tylko w sobotę.
import requests
import json
import subprocess
import os
import sys
from dotenv import load_dotenv

load_dotenv()
CENTRAL_REPORT_URL = os.getenv("REPORT_URL")
if not CENTRAL_REPORT_URL:
    raise RuntimeError("REPORT_URL nie ustawiony w .env")

def fetch_and_save_headers():
    headers = {
        "User-Agent": "Mozilla/5.0 (EasterEggHunter 1.0)"
    }
    resp = requests.get(CENTRAL_REPORT_URL, headers=headers)
    data = {
        "status_code": resp.status_code,
        "url": resp.url,
        "headers": dict(resp.headers),
        "body": resp.text
    }
    with open("headers-httpx.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    xegg = resp.headers.get("x-easteregg")
    if xegg:
        print("🥚 ZNALEZIONO X-EASTER-EGG:", xegg)
    else:
        print("Brak nagłówka x-easteregg")
    # Flaga w body?
    if "FLG:" in resp.text:
        print("🎯 FLAGA w odpowiedzi JSON:\n", resp.text)

def choose_and_run_zadanie():
    # Szukaj plików zadN.py w bieżącym katalogu
    files = [f for f in os.listdir(".") if f.startswith("zad") and f.endswith(".py")]
    if not files:
        print("Nie znaleziono żadnych plików zadN.py w bieżącym katalogu.")
        sys.exit(1)
    print("Dostępne zadania:")
    for idx, fname in enumerate(files):
        print(f"{idx+1}. {fname}")
    while True:
        try:
            choice = int(input("Podaj numer zadania do uruchomienia: "))
            if 1 <= choice <= len(files):
                break
            else:
                print("Niepoprawny numer, spróbuj jeszcze raz.")
        except Exception:
            print("Niepoprawna wartość, spróbuj jeszcze raz.")
    zadanie_file = files[choice-1]
    print(f"Uruchamiam {zadanie_file} ...")
    ret = subprocess.run([sys.executable, zadanie_file])
    if ret.returncode != 0:
        print(f"Błąd podczas wykonywania {zadanie_file}. Kod wyjścia: {ret.returncode}")
        sys.exit(1)

def send_answer_json():
    # Wysyła plik answer.json jeśli istnieje
    if not os.path.isfile("answer.json"):
        print("Brak pliku answer.json, nie wysyłam nic do centrali.")
        return
    with open("answer.json") as f:
        payload = json.load(f)
    print("Wysyłam zawartość answer.json do centrali...")
    resp = requests.post(CENTRAL_REPORT_URL, json=payload)
    print("Status odpowiedzi:", resp.status_code)
    try:
        print("Odpowiedź JSON:", resp.json())
    except Exception:
        print("Odpowiedź tekstowa:", resp.text)

def main():
    print("[1/3] Pobieram nagłówki i zapisuję headers-httpx.json...")
    fetch_and_save_headers()
    print("[2/3] Uruchamiam wybrane zadanie zadN.py...")
    choose_and_run_zadanie()
    print("[3/3] Wysyłam answer.json do centrali (jeśli istnieje)...")
    send_answer_json()

if __name__ == "__main__":
    main()
