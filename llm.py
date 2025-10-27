#!/usr/bin/env python3
from pathlib import Path
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

def summarise_file(file_path, model="phi3"):
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

    # Handle summarise flag
    if "--summarise" in args:
        # Support both spellings
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

        summarise_file(file_path, model=model)
        sys.exit(0)

    # Handle normal question mode
    if "--verbose" in args: args.remove("--verbose")
    if "--raw" in args: args.remove("--raw")
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
