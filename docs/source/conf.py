# Configuration file for the Sphinx documentation builder.

import os
import sys

# ------------------------------------------------------------
# Add project root to sys.path so autodoc can import modules
# ------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../../'))   # project root
sys.path.insert(0, os.path.abspath('../../backend'))  # optional

# ------------------------------------------------------------
# Project information
# ------------------------------------------------------------
project = 'Interpolator Package Documentation'
author = 'xxx'
copyright = '2025, xxx'
release = '0.1.0'

# ------------------------------------------------------------
# General configuration
# ------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode"
]

autosummary_generate = True  # Automatically generate summary pages

templates_path = ['_templates']
exclude_patterns = []

# ------------------------------------------------------------
# Mock heavy imports (so Sphinx can import your modules)
# ------------------------------------------------------------
autodoc_mock_imports = [
    "fastapi",
    "pydantic",
    "starlette",
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "uvicorn"
]

# ------------------------------------------------------------
# HTML Output
# ------------------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

# Custom CSS for better line spacing
def setup(app):
    app.add_css_file('custom.css')
