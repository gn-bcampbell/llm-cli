#!/usr/bin/env python3
import sys
import requests
import warnings
from rich.console import Console
from rich.markdown import Markdown

# Suppress SSL-related warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

console = Console()
OLLAMA_API = "http://localhost:11434/api/generate"

def is_ollama_running():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=1)
        return True
    except requests.ConnectionError:
        return False

def query_ollama(prompt, model="phi3"):
    """Query Ollama's local API directly and return the model output."""
    response = requests.post(
        OLLAMA_API,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()

def main():
    # Parse CLI args
    args = sys.argv[1:]
    if not args:
        console.print("[bold red]Usage:[/bold red] llm [--verbose|--raw] [--model MODEL] <your question>")
        sys.exit(1)

    verbose = "--verbose" in args
    raw = "--raw" in args
    model = "phi3"

    # Remove flags from args
    if "--verbose" in args: args.remove("--verbose")
    if "--raw" in args: args.remove("--raw")
    if "--model" in args:
        try:
            model_index = args.index("--model")
            model = args[model_index + 1]
            del args[model_index:model_index + 2]
        except IndexError:
            console.print("[red]Missing model name after --model[/red]")
            sys.exit(1)

    question = " ".join(args)

    if not is_ollama_running():
        console.print("[bold red]❌ Ollama server not running.[/bold red]")
        console.print("Start it with: [yellow]brew services start ollama[/yellow]")
        sys.exit(1)

    if verbose:
        console.print(f"[bold cyan]🤖 Asking local LLM:[/bold cyan] {question}\n")

    prompt = (
        "You are a concise command-line assistant. "
        "Return clear answers and minimal explanations, using code blocks where helpful.\n\n"
        f"Question: {question}\n"
    )

    try:
        answer = query_ollama(prompt, model=model)
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