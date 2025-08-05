#!/bin/env python
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Match, Optional, Tuple, TypeVar

import fitz  # PyMuPDF
import ollama

R = TypeVar("R")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create renamed symlinks for PDFs.")
    parser.add_argument("directory", type=Path, help="Input directory with PDFs")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where symlinks with new names will be created",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating symlinks",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing symlinks in output directory",
    )

    args = parser.parse_args()
    rename_pdfs(
        input_dir=args.directory,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        recursive=args.recursive,
        force=args.force,
    )


def rename_pdfs(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    recursive: bool = False,
    force: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in input_dir.glob(pattern) if f.is_file() and f.suffix.lower() == ".pdf"
    )
    counter = 1

    for path in files:
        print(f"processing: {path}")

        title = extract_document_title(path)
        if not title and "?" in path.name:
            print(f"skipping {path}")
            continue

        base = title or f"?-{counter}"
        if not title:
            counter += 1

        symlink_name = f"{base}{path.suffix}"
        symlink_path = output_dir / symlink_name

        suffix = 1
        while symlink_path.exists() and not force:
            symlink_path = output_dir / f"{base}_{suffix}{path.suffix}"
            suffix += 1

        if dry_run:
            print(f"[dry-run] Would symlink: {symlink_path} → {path}")
        else:
            try:
                if symlink_path.exists() and force:
                    symlink_path.unlink()
                symlink_path.symlink_to(path.resolve())
                print(f"Created symlink: {symlink_path} → {path}")
            except Exception as e:
                print(f"[error] Failed to create symlink for {path}: {e}")


def extract_document_title(path: Path) -> Optional[str]:
    try:
        doc = fitz.open(str(path))
        raw = extract_text_preview(path)
        if not raw:
            print(f"{path} has no preview (probably a PDF without readable text)")
            return None

        raw = collapse_same_whitespace(raw)
        raw = remove_superfluous_lines(raw)
        title = generate_title(raw)
        return title if title else None

    except Exception as e:
        print(f"[PDF title extraction error] {e}")
        return None
    finally:
        doc.close()


def generate_title(text: str, model: str = "llama3.2:1b") -> Optional[str]:
    prompt = """
    What is the title of this document? If you cannot find it in the
    following extract, then what do you think a good title would be?

    Avoid filler words and verbose phrasing. Avoid generic terms and
    be specific. Avoid phrases like "book review" which don't really
    give any specifics.

    Reply only with the phrase on a single line and nothing more.
    """
    content = f"Document contents: \n\n {text.strip()}"
    try:
        response, duration = measure_time(
            lambda: ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ],
                options={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repeat_penalty": 1.1,
                },
            )
        )
        assert response.message.content == ""
        title = slugify(response.message.content or "")
        title = ensure_title_restrictions(title)
        print(f"LLM output (generated in {duration:.2f}s): {title}")
        return title

    except ollama.ResponseError as e:
        print(f"[Ollama error] {e}")
        return None


def ensure_title_restrictions(title: str, word_limit: int = 10) -> str:
    words = title.split("-")
    words = words[:word_limit]
    return "-".join(words)


def extract_text_preview(path: Path, min_chars: int = 1000, max_pages: int = 10) -> str:
    try:
        doc = fitz.open(str(path))
        text_chunks = []
        total_chars = 0

        for page in doc[:max_pages]:
            text = page.get_text("text")
            cleaned = text.strip()
            if cleaned:
                text_chunks.append(cleaned)
                total_chars += len(cleaned)
                if total_chars >= min_chars:
                    break

        return "\n\n".join(text_chunks)
    except Exception as e:
        print(f"[PDF text extraction error] {e}")
        return ""


def remove_superfluous_lines(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if is_just_punctuation(line) or line.isdigit():
            continue
        lines.append(line)
    return "\n".join(lines)


def collapse_same_whitespace(text: str) -> str:
    def _replace(match: Match[str]) -> str:
        whitespace = match.group(0)
        if "\n" in whitespace:
            return "\n"
        if "\t" in whitespace:
            return "\t"
        return " "

    text = re.sub(r"\s{2,}", _replace, text)
    return text


def is_just_punctuation(s: str) -> bool:
    return bool(s) and all(unicodedata.category(c).startswith("P") for c in s)


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower().replace(" ", "-")


def measure_time(func: Callable[..., R], *args: Any, **kwargs: Any) -> Tuple[R, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


if __name__ == "__main__":
    main()
