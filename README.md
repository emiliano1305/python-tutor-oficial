# Tutor de programación Python

Aplicación Streamlit en español que busca pasajes en una selección de la **documentación oficial de Python 3** y puede generar explicaciones/código mediante OpenAI cuando se configura una clave válida. No usa PDF, no ejecuta código y no consulta otras fuentes.

## Qué incluye

- Tutorial oficial: sintaxis inicial, control de flujo, estructuras de datos, funciones, módulos, archivos, excepciones, clases, entornos virtuales y biblioteca estándar.
- Referencia formal del lenguaje: modelo de datos, expresiones, importación y sentencias.
- Temas de biblioteca estándar: `pathlib`, `json`, `typing`, `asyncio`, `unittest`, concurrencia, `collections`, `itertools` y `contextlib`.
- Recuperación local de pasajes TF-IDF con enlaces directos a `docs.python.org`.
- Acepta consultas en español y expande vocabulario técnico frecuente para buscar en la documentación en inglés.
- Si no hay clave API válida, funciona como buscador didáctico y muestra los pasajes; no inventa una respuesta generativa.
- Si OpenAI devuelve un error de autenticación, no imprime la clave ni el mensaje crudo que podría contenerla; igualmente deja ver fuentes recuperadas.
- Restringe las respuestas generativas a programación Python y a la evidencia recuperada.

## Fuente y límites

La fuente elegida es `https://docs.python.org/3/` (tutorial, referencia del lenguaje y partes de la biblioteca estándar). El índice se descarga al iniciar y se conserva en caché por 24 horas. La aplicación no busca toda la web y puede abstenerse si la pregunta no coincide con los pasajes indexados. La documentación oficial está mayormente en inglés; con una clave válida, OpenAI normaliza la consulta y redacta en español usando la evidencia obtenida.

El tutorial oficial aclara que está pensado para personas que ya conocen conceptos básicos de programación. Por ello, algunos fundamentos generales de programación podrían requerir una fuente adicional, pero el tutor no la incorporará mientras se mantenga el requisito de usar únicamente documentación oficial de Python.

## OpenAI: opcional para respuestas generadas

Para probar la búsqueda de documentación no se necesita una clave. Para generar explicaciones o código se requiere una **clave válida de OpenAI API** y el modelo configurado. La clave se guarda solo en Streamlit Secrets, nunca en GitHub:

```toml
OPENAI_API_KEY = "tu_clave_api"
OPENAI_MODEL = "gpt-4o-mini"
```

Cada consulta generativa puede hacer hasta dos solicitudes: normalizar la búsqueda y redactar la respuesta. Considerá eso antes de compartir públicamente la aplicación. No compartas tu clave en chats, capturas ni commits.

## Ejecutar localmente

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Si querés respuestas generativas, completá `OPENAI_API_KEY` localmente en `.env`. Sin esa variable, la app mostrará resultados oficiales recuperados sin redactar una respuesta nueva.

## Desplegar en Streamlit Community Cloud

- Repositorio: tu repositorio GitHub.
- Rama: `main`.
- Archivo principal: `app.py`.
- Versión sugerida: Python 3.12.
- Para generación con IA: agregar `OPENAI_API_KEY` y `OPENAI_MODEL` en **App settings → Secrets**; no en GitHub.

## Estructura

```text
python-tutor-web/
├── app.py
├── core.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── tests/test_core.py
```

## Pruebas

```bash
pytest -q
```
