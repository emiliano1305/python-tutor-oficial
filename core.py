from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = "https://docs.python.org/3/"

# Fuentes oficiales seleccionadas: tutorial, referencia del lenguaje y biblioteca estándar.
SOURCE_PATHS = [
    "tutorial/index.html",
    "tutorial/introduction.html",
    "tutorial/controlflow.html",
    "tutorial/datastructures.html",
    "tutorial/modules.html",
    "tutorial/inputoutput.html",
    "tutorial/errors.html",
    "tutorial/classes.html",
    "tutorial/stdlib.html",
    "tutorial/stdlib2.html",
    "tutorial/venv.html",
    "reference/index.html",
    "reference/datamodel.html",
    "reference/executionmodel.html",
    "reference/import.html",
    "reference/expressions.html",
    "reference/simple_stmts.html",
    "reference/compound_stmts.html",
    "library/functions.html",
    "library/stdtypes.html",
    "library/pathlib.html",
    "library/json.html",
    "library/typing.html",
    "library/asyncio.html",
    "library/unittest.html",
    "library/concurrent.futures.html",
    "library/collections.html",
    "library/itertools.html",
    "library/contextlib.html",
]

SPANISH_TERMS = {
    "variable": "variables names assignment values",
    "variables": "variables names assignment values",
    "bucle": "loops for while break continue iteration",
    "bucles": "loops for while break continue iteration",
    "ciclo": "loops for while iteration",
    "ciclos": "loops for while iteration",
    "condicional": "if statements conditional expressions control flow",
    "condicionales": "if statements conditional expressions control flow",
    "función": "functions def arguments return parameters",
    "funciones": "functions def arguments return parameters",
    "lista": "list data structures sequence append",
    "listas": "list data structures sequence append",
    "diccionario": "dictionary dict mapping keys values",
    "diccionarios": "dictionary dict mapping keys values",
    "tupla": "tuple sequence immutable",
    "tuplas": "tuple sequence immutable",
    "conjunto": "set collection unique elements",
    "conjuntos": "set collection unique elements",
    "clase": "class object oriented programming methods",
    "clases": "class object oriented programming methods",
    "objeto": "object data model type",
    "objetos": "object data model type",
    "error": "errors exceptions traceback try except",
    "errores": "errors exceptions traceback try except",
    "excepción": "exception errors try except raise",
    "excepciones": "exception errors try except raise",
    "módulo": "module import package",
    "módulos": "module import package",
    "paquete": "package modules import",
    "paquetes": "package modules import",
    "archivo": "file input output open read write",
    "archivos": "file input output open read write",
    "leer": "read file input output",
    "escribir": "write file input output",
    "texto": "string text str methods",
    "cadena": "string text str methods",
    "cadenas": "string text str methods",
    "asíncrono": "async await asyncio asynchronous coroutines",
    "asincrono": "async await asyncio asynchronous coroutines",
    "asincrónica": "async await asyncio asynchronous coroutines",
    "tipado": "typing type hints annotations",
    "tipos": "types type hints data model",
    "prueba": "unittest testing test cases",
    "pruebas": "unittest testing test cases",
    "concurrencia": "concurrent futures threading multiprocessing asyncio",
    "iterador": "iterator iterable yield generator",
    "iteradores": "iterator iterable yield generator",
    "generador": "generator yield iterator",
    "generadores": "generator yield iterator",
    "comprensión": "comprehensions list dict set",
    "comprensiones": "comprehensions list dict set",
    "importar": "import modules package",
    "importación": "import modules package",
    "entorno": "virtual environment venv packages",
    "entornos": "virtual environment venv packages",
}


@dataclass(frozen=True)
class Passage:
    title: str
    url: str
    text: str


def expand_spanish_query(query: str) -> str:
    """Añade equivalentes ingleses de vocabulario Python frecuente en consultas en español."""
    normalized = query.lower()
    expansions: list[str] = []
    for spanish, english in SPANISH_TERMS.items():
        if spanish in normalized:
            expansions.append(english)
    return f"{query} {' '.join(expansions)}".strip()


def _split_words(text: str, size: int = 170, overlap: int = 35) -> Iterable[str]:
    words = re.findall(r"\S+", text)
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        part = " ".join(words[start : start + size]).strip()
        if part:
            yield part
        if start + size >= len(words):
            break


def _extract_page(html: str, url: str) -> list[Passage]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("div.body") or soup.select_one("main") or soup.body
    if root is None:
        return []

    for node in root.select("script, style, nav, footer, .related, .sphinxsidebar, .toctree-wrapper"):
        node.decompose()

    page_title_node = soup.find("h1") or soup.title
    page_title = page_title_node.get_text(" ", strip=True) if page_title_node else url.rsplit("/", 1)[-1]
    page_title = re.sub(r"\s*¶\s*$", "", page_title).strip()

    passages: list[Passage] = []
    current_heading = page_title
    current_text: list[str] = []

    def flush() -> None:
        nonlocal current_text, current_heading
        text = re.sub(r"\s+", " ", " ".join(current_text)).strip()
        if text:
            for part in _split_words(text):
                passages.append(Passage(current_heading, url, part))
        current_text = []

    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li", "dt", "dd"]):
        if element.name in {"h1", "h2", "h3", "h4"}:
            flush()
            heading = element.get_text(" ", strip=True)
            if heading:
                current_heading = re.sub(r"\s*¶\s*$", "", heading).strip()
        else:
            text = element.get_text(" ", strip=True)
            if text:
                current_text.append(text)
    flush()
    return passages


def fetch_python_docs() -> list[Passage]:
    """Descarga solo una lista controlada de páginas desde docs.python.org."""
    session = requests.Session()
    session.headers.update({"User-Agent": "PythonTutorEducationalApp/1.0 (official docs reader)"})
    all_passages: list[Passage] = []
    failures: list[str] = []

    for path in SOURCE_PATHS:
        url = BASE + path
        try:
            response = session.get(url, timeout=(5, 20))
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", ""):
                failures.append(path)
                continue
            all_passages.extend(_extract_page(response.text, url))
            time.sleep(0.03)
        except requests.RequestException:
            failures.append(path)

    if not all_passages:
        details = ", ".join(failures[:4])
        raise RuntimeError(f"No se pudieron descargar las páginas oficiales de Python. Fallos: {details}")
    return all_passages


class PythonDocsIndex:
    def __init__(self, passages: list[Passage]):
        if not passages:
            raise ValueError("No hay pasajes oficiales para indexar.")
        self.passages = passages
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=200_000,
        )
        self.matrix = self.vectorizer.fit_transform(
            [f"{p.title} {p.text}" for p in passages]
        )

    def search(self, query: str, top_k: int = 5) -> list[tuple[Passage, float]]:
        query = query.strip()
        if not query:
            return []
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1][: max(1, top_k)]
        return [(self.passages[i], float(scores[i])) for i in ranked if scores[i] > 0]


def make_evidence(found: list[tuple[Passage, float]]) -> tuple[str, dict[str, Passage]]:
    refs: dict[str, Passage] = {}
    blocks: list[str] = []
    for index, (passage, score) in enumerate(found, start=1):
        source_id = f"D{index}"
        refs[source_id] = passage
        blocks.append(
            f"[{source_id}] {passage.title}\nURL: {passage.url}\n"
            f"Relevancia de búsqueda: {score:.3f}\n{passage.text}"
        )
    return "\n\n---\n\n".join(blocks), refs


def validate_citations(answer: str, allowed_ids: set[str]) -> str:
    pattern = re.compile(r"\[(D\d+)\]")

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1) in allowed_ids else "[referencia no verificada]"

    return pattern.sub(replace, answer)
