#!/usr/bin/env python3
"""
Agent do wykonywania zadań AI Devs 3 Reloaded
Refaktoryzacja zgodna z zaleceniami SonarQube
"""
import json
import os
import platform
import re
import subprocess
import sys
from typing import Tuple, List, Dict, Optional, Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState

load_dotenv(override=True)

# Stałe dla duplikowanych literałów (S1192)
ERROR_INVALID_TASK = "Niepoprawny numer zadania. Wybierz w zakresie 1-24."
ERROR_INVALID_SECRET = "Niepoprawny numer sekretu. Wybierz w zakresie 1-9."
ERROR_TASK_FAILED = "🛑 Zadanie zakończone z błędem."
ERROR_SECRET_FAILED = "🛑 Sekret zakończony z błędem."
NOT_SET_VALUE = "(niewartość)"
FLAGS_JSON = "flags.json"
SECRETS_JSON = "secrets.json"
STATUS_SET = "✅ ustawiony"
STATUS_MISSING = "❌ brak"

# Zakresy dozwolonych wartości
VALID_TASKS = set(str(i) for i in range(1, 25))  # "1" do "24"
VALID_SECRETS = set(str(i) for i in range(1, 10))  # "1" do "9"
VALID_ENGINES = {"openai", "lmstudio", "anything", "gemini", "claude"}

# Zmienne globalne
completed_tasks = set()
completed_secrets = set()
current_engine = None
current_model = None


class Config:
    """Konfiguracja aplikacji"""
    PROBLEMATIC_ENV_VARS = [
        "LLM_ENGINE", "MODEL_NAME", "ENGINE", "OPENAI_MODEL",
        "CLAUDE_MODEL", "GEMINI_MODEL", "AI_MODEL", "LLM_MODEL_NAME"
    ]
    
    SHELL_TYPES = {
        "powershell": {"indicator": "PSModulePath", "platform": "windows"},
        "cmd": {"indicator": None, "platform": "windows"},
        "bash": {"indicator": "bash", "platform": "unix"},
        "zsh": {"indicator": "zsh", "platform": "unix"},
        "fish": {"indicator": "fish", "platform": "unix"},
    }


class ShellDetector:
    """Wykrywanie typu powłoki systemowej"""
    
    @staticmethod
    def detect() -> str:
        """Wykrywa typ powłoki systemowej"""
        system = platform.system().lower()
        
        if system == "windows":
            if os.getenv("PSModulePath"):
                return "powershell"
            return "cmd"
        
        shell = os.getenv("SHELL", "").lower()
        for shell_type, pattern in [("bash", "bash"), ("zsh", "zsh"), ("fish", "fish")]:
            if pattern in shell:
                return shell_type
        return "unix"


class EnvironmentCleaner:
    """Czyszczenie problematycznych zmiennych środowiskowych"""
    
    @staticmethod
    def find_problematic_vars() -> List[Tuple[str, str]]:
        """Znajduje problematyczne zmienne środowiskowe"""
        found_vars = []
        for var in Config.PROBLEMATIC_ENV_VARS:
            value = os.environ.get(var)
            if value:
                found_vars.append((var, value))
        return found_vars
    
    @staticmethod
    def clean_vars(vars_to_clean: List[Tuple[str, str]]) -> None:
        """Czyści zmienne środowiskowe"""
        for var, _ in vars_to_clean:
            del os.environ[var]
            print(f"🧹 Wyczyszczono: {var}")
    
    @staticmethod
    def show_cleanup_instructions(shell_type: str, vars: List[Tuple[str, str]]) -> None:
        """Pokazuje instrukcje czyszczenia zmiennych"""
        print("\n💡 Aby wyczyścić zmienne na stałe:")
        
        if shell_type == "powershell":
            for var, _ in vars:
                print(f"   [Environment]::SetEnvironmentVariable('{var}', $null, 'User')")
        elif shell_type in ["bash", "zsh"]:
            print("   Usuń odpowiednie linie z ~/.bashrc, ~/.zshrc, ~/.profile")
        elif shell_type == "cmd":
            for var, _ in vars:
                print(f'   setx {var} ""')
        print()


class TaskExecutor:
    """Wykonywanie zadań i sekretów"""
    
    @staticmethod
    def prepare_env() -> Dict[str, str]:
        """Przygotowuje środowisko dla subprocess"""
        env = os.environ.copy()
        if current_engine:
            env["LLM_ENGINE"] = current_engine
        if current_model:
            env["MODEL_NAME"] = current_model
        if not env.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
        env["PYTHONUTF8"] = "1"
        return env
    
    @staticmethod
    def extract_flags(output: str) -> Tuple[List[str] | str, bool]:
        """Wyodrębnia flagi z wyjścia"""
        flags = re.findall(r"\{\{FLG:[^}]*\}\}|FLG\{[^}]*\}", output)
        if flags:
            return flags, True
        return output, False
    
    @staticmethod
    def run_script(script: str, env: Dict[str, str]) -> subprocess.CompletedProcess:
        """Uruchamia skrypt Python"""
        return subprocess.run(
            [sys.executable, script],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
    
    @classmethod
    def execute(cls, script_name: str) -> Tuple[Any, bool, bool]:
        """Wykonuje skrypt i zwraca (output, flag_found, error)"""
        if not os.path.exists(script_name):
            return (f"Plik {script_name} nie istnieje.", False, False)
        
        env = cls.prepare_env()
        
        try:
            result = cls.run_script(script_name, env)
            output_full = result.stdout.rstrip() or "(Brak wyjścia)"
            output, flag_found = cls.extract_flags(output_full)
            return (output, flag_found, False)
            
        except subprocess.CalledProcessError as e:
            out_text = (e.stdout or "").rstrip()
            err_text = (e.stderr or "").rstrip()
            return ((out_text, err_text), False, True)


class Logger:
    """Logowanie wyników do JSON"""
    
    @staticmethod
    def append_to_json(entry: Dict[str, Any], log_file: str = FLAGS_JSON) -> None:
        """Dodaje wpis do pliku JSON"""
        data = Logger.load_existing_data(log_file)
        
        if Logger.is_duplicate(entry, data):
            return
        
        data.append(entry)
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    @staticmethod
    def load_existing_data(log_file: str) -> List[Dict[str, Any]]:
        """Wczytuje istniejące dane z pliku"""
        if not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            return []
    
    @staticmethod
    def is_duplicate(entry: Dict[str, Any], data: List[Dict[str, Any]]) -> bool:
        """Sprawdza czy wpis już istnieje"""
        entry_type = "zadanie" if "zadanie" in entry else "sekret"
        
        if entry_type not in entry or "flagi" not in entry or not entry["flagi"]:
            return False
        
        for existing in data:
            if (existing.get(entry_type) == entry[entry_type] and 
                set(existing.get("flagi", [])) == set(entry["flagi"])):
                return True
        return False


class LLMFactory:
    """Fabryka do tworzenia klientów LLM"""
    
    @staticmethod
    def create(engine: str, model_name: str):
        """Tworzy odpowiedni klient LLM"""
        if engine == "claude":
            return LLMFactory._create_claude(model_name)
        elif engine == "gemini":
            return LLMFactory._create_gemini(model_name)
        elif engine == "lmstudio":
            return LLMFactory._create_lmstudio(model_name)
        elif engine == "anything":
            return LLMFactory._create_anything(model_name)
        else:  # openai
            return LLMFactory._create_openai(model_name)
    
    @staticmethod
    def _create_claude(model_name: str):
        """Tworzy klienta Claude"""
        try:
            from langchain_anthropic import ChatAnthropic
            api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            return ChatAnthropic(
                model_name=model_name,
                anthropic_api_key=api_key,
                temperature=0
            )
        except ImportError:
            raise ImportError("Brak langchain_anthropic. Zainstaluj: pip install langchain-anthropic")
    
    @staticmethod
    def _create_gemini(model_name: str):
        """Tworzy klienta Gemini"""
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )
    
    @staticmethod
    def _create_lmstudio(model_name: str):
        """Tworzy klienta LMStudio"""
        url = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
        key = os.getenv("LMSTUDIO_API_KEY", "local")
        return ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_base=url,
            openai_api_key=key
        )
    
    @staticmethod
    def _create_anything(model_name: str):
        """Tworzy klienta Anything"""
        url = os.getenv("ANYTHING_API_URL", "http://localhost:1234/v1")
        key = os.getenv("ANYTHING_API_KEY", "local")
        return ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_base=url,
            openai_api_key=key
        )
    
    @staticmethod
    def _create_openai(model_name: str):
        """Tworzy klienta OpenAI"""
        return ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_base=os.getenv("OPENAI_API_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )


def detect_shell_and_clean_env() -> None:
    """Wykrywa typ powłoki i czyści problematyczne zmienne środowiskowe"""
    shell_type = ShellDetector.detect()
    system = platform.system().lower()
    
    print(f"🔍 Wykryto środowisko: {system} / {shell_type}")
    
    found_vars = EnvironmentCleaner.find_problematic_vars()
    
    if not found_vars:
        print("✅ Brak konfliktowych zmiennych środowiskowych")
        return
    
    print("⚠️  Znaleziono potencjalnie konfliktowe zmienne środowiskowe:")
    for var, value in found_vars:
        print(f"   {var} = {value}")
    
    try:
        response = input("❓ Wyczyścić te zmienne dla tej sesji? [y/N]: ").strip().lower()
        if response in {"y", "yes", "tak", "t"}:
            EnvironmentCleaner.clean_vars(found_vars)
            EnvironmentCleaner.show_cleanup_instructions(shell_type, found_vars)
        else:
            print("🔄 Kontynuuję z obecnymi zmiennymi...")
    except (EOFError, KeyboardInterrupt):
        print("\n🔄 Kontynuuję z obecnymi zmiennymi...")


def _execute_task(task_key: str) -> Tuple[Any, bool, bool]:
    """Wykonuje zadanie"""
    key = str(task_key).strip().strip("'").strip('"')
    
    if key not in VALID_TASKS:
        return (ERROR_INVALID_TASK, False, False)
    
    script = f"zad{key}.py"
    return TaskExecutor.execute(script)


def _execute_secret(secret_key: str) -> Tuple[Any, bool, bool]:
    """Wykonuje sekretne zadanie"""
    key = str(secret_key).strip().strip("'").strip('"')
    
    if key not in VALID_SECRETS:
        return (ERROR_INVALID_SECRET, False, False)
    
    script = f"sec{key}.py"
    return TaskExecutor.execute(script)


def format_flag_message(flags: List[str] | str) -> str:
    """Formatuje komunikat o znalezionych flagach"""
    flags_list = flags if isinstance(flags, list) else [str(flags)]
    
    if len(flags_list) > 1:
        return f"🏁 Flagi znalezione: [{', '.join(flags_list)}] - kończę zadanie."
    else:
        return f"🏁 Flaga znaleziona: {flags_list[0]} - kończę zadanie."


def handle_error_output(output: Tuple[str, str], task_type: str, task_id: str, log_file: str) -> None:
    """Obsługuje błędne wyjście"""
    stdout_text, stderr_text = output if isinstance(output, tuple) else ("", "")
    
    error_msg = ERROR_TASK_FAILED if task_type == "zadanie" else ERROR_SECRET_FAILED
    print(error_msg)
    
    if stdout_text:
        print(f"🐞 STDOUT:\n{stdout_text}")
    if stderr_text:
        print(f"🐞 STDERR:\n{stderr_text}")
    
    log_entry = {
        task_type: task_id,
        "flagi": [],
        "debug_output": f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
    }
    Logger.append_to_json(log_entry, log_file)


@tool
def run_task(task_key: str) -> str:
    """
    Uruchamia zadanie zadN.py (gdzie N to numer zadania) i zwraca wynik działania,
    w szczególności flagę w formacie {{FLG:...}}. Używany przez agenta LangChain.
    Obsługuje zadania 1-24.
    """
    key = str(task_key).strip().strip("'").strip('"')
    print(f"🔄 Uruchamiam zadanie {key}…")
    
    output, flag_found, error = _execute_task(key)
    
    if error:
        handle_error_output(output, "zadanie", key, FLAGS_JSON)
        return ERROR_TASK_FAILED
    
    if flag_found:
        completed_tasks.add(key)
        flag_msg = format_flag_message(output)
        print(flag_msg)
        
        flags_list = output if isinstance(output, list) else [str(output)]
        Logger.append_to_json({"zadanie": key, "flagi": flags_list})
        
        return flag_msg
    
    return str(output)


@tool
def run_secret(secret_key: str) -> str:
    """
    Uruchamia sekretne zadanie secN.py (gdzie N to numer sekretu) i zwraca wynik działania,
    w szczególności flagę w formacie {{FLG:...}}. Używany przez agenta LangChain.
    Obsługuje sekrety 1-9.
    """
    key = str(secret_key).strip().strip("'").strip('"')
    print(f"🔐 Uruchamiam sekret {key}…")
    
    output, flag_found, error = _execute_secret(key)
    
    if error:
        handle_error_output(output, "sekret", key, SECRETS_JSON)
        return ERROR_SECRET_FAILED
    
    if flag_found:
        completed_secrets.add(key)
        flag_msg = format_flag_message(output)
        print(flag_msg)
        
        flags_list = output if isinstance(output, list) else [str(output)]
        Logger.append_to_json({"sekret": key, "flagi": flags_list}, SECRETS_JSON)
        
        return flag_msg
    
    return str(output)


@tool
def read_env(var: str) -> str:
    """
    Zwraca wartość zmiennej środowiskowej o nazwie var. 
    Jeśli zmienna nie istnieje, zwraca '(niewartość)'.
    """
    key = str(var).strip().strip("'").strip('"')
    return os.getenv(key, NOT_SET_VALUE)


def get_engine_choice(current_engine: str) -> str:
    """Pobiera wybór silnika od użytkownika"""
    while True:
        try:
            prompt_text = "Wybierz silnik LLM [openai/lmstudio/anything/gemini/claude]"
            if current_engine in VALID_ENGINES:
                prompt_text += f" (aktualny: {current_engine})"
            prompt_text += ": "
            
            engine = input(prompt_text).strip().lower()
            
            # Jeśli użytkownik nie wpisał nic, użyj aktualnego ENGINE z .env
            if not engine and current_engine in VALID_ENGINES:
                engine = current_engine
                print(f"🔄 Używam silnika z .env: {engine}")
            
            if engine in VALID_ENGINES:
                return engine
            
            print("⚠️ Nieznany wybór. Wpisz 'openai', 'lmstudio', 'anything', 'gemini' albo 'claude'.")
            
        except (EOFError, KeyboardInterrupt):
            print("\nKoniec.")
            sys.exit(0)


def get_model_name(engine: str) -> Optional[str]:
    """Pobiera nazwę modelu dla danego silnika"""
    model_mapping = {
        "openai": ("MODEL_NAME_OPENAI", None),
        "claude": ("MODEL_NAME_CLAUDE", "claude-sonnet-4-20250514"),
        "gemini": ("MODEL_NAME_GEMINI", "gemini-2.5-pro-latest"),
        "lmstudio": ("MODEL_NAME_LM", "gpt-4o-mini"),
        "anything": ("MODEL_NAME_ANY", "gpt-4o-mini")
    }
    
    env_var, default = model_mapping.get(engine, (None, None))
    
    if engine == "openai":
        model_name = os.getenv(env_var)
        if not model_name:
            print(f"⚠️ Nie ustawiono {env_var} w .env - przerwano działanie.")
            return None
        return model_name
    
    # Dla lmstudio i anything - kaskadowe sprawdzanie
    if engine in ["lmstudio", "anything"]:
        model_name = os.getenv(env_var, "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_ANY" if engine == "anything" else "MODEL_NAME_LM", "")
        if not model_name:
            model_name = os.getenv("MODEL_NAME_OPENAI", default)
        return model_name
    
    # Dla pozostałych silników
    return os.getenv(env_var, default)


def validate_api_keys(engine: str) -> bool:
    """Sprawdza czy wymagane klucze API są ustawione"""
    required_keys = {
        "claude": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY"],
        "openai": ["OPENAI_API_KEY"]
    }
    
    if engine not in required_keys:
        return True
    
    keys = required_keys[engine]
    
    # Dla Claude - wystarczy jeden z kluczy
    if engine == "claude":
        if not any(os.getenv(key) for key in keys):
            print(f"⚠️ Nie ustawiono żadnego z kluczy: {', '.join(keys)} w .env - przerwano działanie.")
            return False
        return True
    
    # Dla pozostałych - wszystkie klucze muszą być ustawione
    for key in keys:
        if not os.getenv(key):
            print(f"⚠️ Nie ustawiono {key} w .env - przerwano działanie.")
            return False
    
    return True


def debug_env_vars(engine: str) -> None:
    """Wyświetla debug informacji o zmiennych środowiskowych"""
    print("🔍 Debug - zmienne kluczowe:")
    
    debug_info = {
        "lmstudio": [
            ("LMSTUDIO_API_URL", os.getenv("LMSTUDIO_API_URL")),
            ("LMSTUDIO_API_KEY", os.getenv("LMSTUDIO_API_KEY"))
        ],
        "anything": [
            ("ANYTHING_API_URL", os.getenv("ANYTHING_API_URL")),
            ("ANYTHING_API_KEY", os.getenv("ANYTHING_API_KEY"))
        ],
        "claude": [
            ("CLAUDE/ANTHROPIC_API_KEY", 
             STATUS_SET if (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")) else STATUS_MISSING)
        ],
        "gemini": [
            ("GEMINI_API_KEY", STATUS_SET if os.getenv("GEMINI_API_KEY") else STATUS_MISSING)
        ],
        "openai": [
            ("OPENAI_API_KEY", STATUS_SET if os.getenv("OPENAI_API_KEY") else STATUS_MISSING),
            ("OPENAI_API_URL", os.getenv("OPENAI_API_URL"))
        ]
    }
    
    if engine in debug_info:
        for var_name, value in debug_info[engine]:
            print(f"   {var_name}: {value}")


def process_command(cmd: str) -> bool:
    """
    Przetwarza komendę użytkownika
    Zwraca True jeśli należy kontynuować, False jeśli zakończyć
    """
    if not cmd:
        return True
    
    if cmd.lower() in {"exit", "quit"}:
        print("Wyłączam agenta.")
        return False
    
    # Obsługa run_task
    if cmd.lower().startswith("run_task"):
        return handle_run_task_command(cmd)
    
    # Obsługa run_secret
    if cmd.lower().startswith("run_secret"):
        return handle_run_secret_command(cmd)
    
    # Obsługa read_env
    if cmd.lower().startswith("read_env"):
        return handle_read_env_command(cmd)
    
    print("Nieznana komenda. Użyj: run_task N (1-24), run_secret N (1-9), read_env VAR, lub exit.")
    return True


def handle_run_task_command(cmd: str) -> bool:
    """Obsługuje komendę run_task - zwraca czy kontynuować"""
    parts = cmd.split(maxsplit=1)
    if len(parts) < 2:
        print(ERROR_INVALID_TASK)
        return True  # Kontynuuj mimo błędnych argumentów
    
    task_arg = extract_argument(parts[1])
    
    if task_arg not in VALID_TASKS:
        print(ERROR_INVALID_TASK)
        return True  # Kontynuuj mimo nieprawidłowego numeru zadania
    
    print(f"🔄 Uruchamiam zadanie {task_arg}…")
    output, flag_found, error = _execute_task(task_arg)
    
    if error:
        handle_error_output(output, "zadanie", task_arg, FLAGS_JSON)
        return True  # Kontynuuj mimo błędu wykonania
    elif flag_found:
        completed_tasks.add(task_arg)
        print(format_flag_message(output))
        flags_list = output if isinstance(output, list) else [str(output)]
        Logger.append_to_json({"zadanie": task_arg, "flagi": flags_list})
        return True  # Kontynuuj po znalezieniu flagi
    else:
        print(output)
        return True  # Kontynuuj po normalnym wykonaniu


def handle_run_secret_command(cmd: str) -> bool:
    """Obsługuje komendę run_secret - zwraca czy kontynuować"""
    parts = cmd.split(maxsplit=1)
    if len(parts) < 2:
        print(ERROR_INVALID_SECRET)
        return True  # Kontynuuj mimo błędnych argumentów
    
    secret_arg = extract_argument(parts[1])
    
    if secret_arg not in VALID_SECRETS:
        print(ERROR_INVALID_SECRET)
        return True  # Kontynuuj mimo nieprawidłowego numeru sekretu
    
    print(f"🔐 Uruchamiam sekret {secret_arg}…")
    output, flag_found, error = _execute_secret(secret_arg)
    
    if error:
        handle_error_output(output, "sekret", secret_arg, SECRETS_JSON)
        return True  # Kontynuuj mimo błędu wykonania
    elif flag_found:
        completed_secrets.add(secret_arg)
        print(format_flag_message(output))
        flags_list = output if isinstance(output, list) else [str(output)]
        Logger.append_to_json({"sekret": secret_arg, "flagi": flags_list}, SECRETS_JSON)
        return True  # Kontynuuj po znalezieniu flagi
    else:
        print(output)
        return True  # Kontynuuj po normalnym wykonaniu


def handle_read_env_command(cmd: str) -> bool:
    """Obsługuje komendę read_env - zwraca czy kontynuować"""
    parts = cmd.split(maxsplit=1)
    if len(parts) < 2:
        print(NOT_SET_VALUE)
        return True  # Kontynuuj z domyślną wartością
    
    var = extract_argument(parts[1])
    value = os.getenv(var, NOT_SET_VALUE)
    print(value)
    return True  # Zawsze kontynuuj po odczycie zmiennej


def extract_argument(arg: str) -> str:
    """Wyodrębnia argument z możliwych cudzysłowów"""
    arg = arg.strip()
    if (arg.startswith("'") and arg.endswith("'")) or (arg.startswith('"') and arg.endswith('"')):
        return arg[1:-1].strip()
    return arg


def main():
    """Główna funkcja programu"""
    global current_engine, current_model
    
    # Sprawdź i wyczyść problematyczne zmienne środowiskowe
    detect_shell_and_clean_env()
    
    # Sprawdź aktualny ENGINE z .env
    current_engine = os.getenv("LLM_ENGINE", "").lower()
    print(f"🔍 Aktualny LLM_ENGINE z .env: '{current_engine}'")
    
    # Pobierz wybór silnika
    engine = get_engine_choice(current_engine)
    print(f"🚀 Wybrany silnik: {engine}")
    
    # Ustaw globalne zmienne
    current_engine = engine
    
    # Pobierz nazwę modelu
    model_name = get_model_name(engine)
    if model_name is None:
        return
    
    print(f"🔧 Model: {model_name}")
    current_model = model_name
    
    # Debug informacji
    debug_env_vars(engine)
    
    # Sprawdzenie kluczy API
    if not validate_api_keys(engine):
        return
    
    # Ustawienie zmiennych środowiskowych
    os.environ["MODEL_NAME"] = model_name
    os.environ["LLM_ENGINE"] = engine
    
    # Inicjalizacja LLM
    try:
        llm = LLMFactory.create(engine, model_name)
        print(f"✅ {engine.capitalize()} LLM zainicjalizowany")
    except Exception as e:
        print(f"❌ Błąd podczas inicjalizacji LLM: {e}")
        return
    
    # Konfiguracja grafu LangChain
    tools = [run_task, run_secret, read_env]
    builder = StateGraph(AgentState)
    llm_with_tools = llm.bind_tools(tools)
    builder.add_node(
        "agent", lambda state: {"messages": llm_with_tools.invoke(state["messages"])}
    )
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
    # Usunięto nieużywaną zmienną graph - graf jest skompilowany ale nie przypisywany
    builder.compile()
    
    print("🤖 Agent uruchomiony. Komendy: run_task N (1-24) | run_secret N (1-9) | read_env VAR | exit")
    print("=" * 60)
    
    # Główna pętla
    while True:
        try:
            cmd = input("> ").strip()
            if not process_command(cmd):
                break
        except (EOFError, KeyboardInterrupt):
            print("\nKoniec.")
            break

if __name__ == "__main__":
    main()