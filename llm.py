#!/usr/bin/env python3
import sys
from pathlib import Path

import requests
import warnings
from rich.console import Console
from rich.markdown import Markdown

# Suppress SSL-related warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

console = Console()
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_TAGS_API = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "phi3:latest"
ALIAS_MAP = {
    "qwen": "qwen3-vl:235b-cloud",
    "deepseek": "deepseek-r1:8b",
    "phi3": "phi3:latest",
}

def resolve_model_name(requested_name):
    """Map alias to actual model name in ollama"""
    if not requested_name:
        return requested_name

    requested_name = requested_name.strip()
    if not requested_name:
        return requested_name

    lowered = requested_name.lower()
    for alias, actual in ALIAS_MAP.items():
        if alias in lowered:
            return actual

    if requested_name in ALIAS_MAP.values():
        return requested_name

    raise ValueError("that model isn't available")

def is_ollama_running():
    try:
        requests.get(OLLAMA_TAGS_API, timeout=1)
        return True
    except requests.ConnectionError:
        return False

def query_ollama(prompt, model=DEFAULT_MODEL):
    """Query Ollama's local API directly and return the model output."""
    response = requests.post(
        OLLAMA_API,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()

def summarise_file(file_path, model=DEFAULT_MODEL):
    """Read a file and summarise its contents using the local LLM."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        console.print(f"[red]File not found:[/red] {file_path}")
        sys.exit(1)

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        console.print(f"[red]Error reading file:[/red] {e}")
        sys.exit(1)

    if len(text.strip()) == 0:
        console.print("[yellow]File is empty, nothing to summarise.[/yellow]")
        sys.exit(1)

    console.print(f"[cyan]📄 Summarising:[/cyan] {file_path}")

    prompt = (
        "You are an expert technical writer, not a shell or programming assistant. "
        "Your only job is to read the following text and produce a short, clear summary "
        "in natural language. Do NOT output code, commands, or instructions. "
        "Respond in plain English prose in 3-5 sentences.\n\n"
        f"--- BEGIN TEXT ---\n{text[:8000]}\n--- END TEXT ---\n\n"
        "Summary:"
    )

    summary = query_ollama(prompt, model=model)
    console.print("\n[bold green]🧠 Summary:[/bold green]\n")
    if "```" in summary:
        console.print(Markdown(summary))
    else:
        console.print(summary)

def main():
    # Parse CLI args
    args = sys.argv[1:]
    
    if not args:
        console.print("[bold red]Usage:[/bold red] llm [--verbose|--raw] [--model MODEL] [--summarise FILE] <your question>")
        sys.exit(1)

    verbose = "--verbose" in args
    raw = "--raw" in args
    model = DEFAULT_MODEL
    model_specified = False

    # Remove flags from args
    # if "--verbose" in args: args.remove("--verbose")
    # if "--raw" in args: args.remove("--raw")
    if "--model" in args:
        try:
            model_index = args.index("--model")
            model = args[model_index + 1]
            del args[model_index:model_index + 2]
            model_specified = True
        except IndexError:
            console.print("[red]Missing model name after --model[/red]")
            sys.exit(1)

    # Handle summarise flag
    if "--summarise" in args:
        flag = "--summarise"
        try:
            idx = args.index(flag)
            file_path = args[idx + 1]
            del args[idx:idx + 2]
        except IndexError:
            console.print(f"[red]Missing file path after {flag}[/red]")
            sys.exit(1)

        if not is_ollama_running():
            console.print("[bold red]❌ Ollama server not running.[/bold red]")
            console.print("Start it with: [yellow]brew services start ollama[/yellow]")
            sys.exit(1)

        try:
            resolved_model = resolve_model_name(model)
        except ValueError as err:
            console.print(f"[bold red]{err}[/bold red]")
            sys.exit(1)

        if resolved_model != model and model_specified:
            console.print(f"[cyan]Using model:[/cyan] {resolved_model} (matched from '{model}')")

        summarise_file(file_path, model=resolved_model)
        sys.exit(0)

    # Handle normal question mode
    # if "--verbose" in args: args.remove("--verbose")
    # if "--raw" in args: args.remove("--raw")
    question = " ".join(args)

    if not is_ollama_running():
        console.print("[bold red]❌ Ollama server not running.[/bold red]")
        console.print("Start it with: [yellow]brew services start ollama[/yellow]")
        sys.exit(1)

    try:
        resolved_model = resolve_model_name(model)
    except ValueError as err:
        console.print(f"[bold red]{err}[/bold red]")
        sys.exit(1)

    if resolved_model != model and model_specified:
        console.print(f"[cyan]Using model:[/cyan] {resolved_model} (matched from '{model}')")

    if verbose:
        console.print(f"[bold cyan]🤖 Asking local LLM ({resolved_model}):[/bold cyan] {question}\n")

    prompt = (
        "You are a concise command-line assistant. "
        "Return clear answers and minimal explanations, using code blocks where helpful.\n\n"
        f"Question: {question}\n"
    )

    try:
        answer = query_ollama(prompt, model=resolved_model)
        if raw:
            print(answer)
        else:
            if "```" in answer:
                console.print(Markdown(answer))
            else:
                console.print(f"[green]{answer}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
