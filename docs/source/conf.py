# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os                                                                                                         
import sys                                                                                                        
sys.path.insert(0, os.path.abspath('../../src'))  

import tomllib

with open("../../pyproject.toml", "rb") as f:
    toml_data = tomllib.load(f)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = f"{toml_data['project']['name']}"
copyright = '2026, Jakub Mandula'
author = f"{toml_data['project']['authors'][0]['name']}"
release = f"v{toml_data['project']['version']}"

# Version-switcher wiring (set by the CI docs jobs). DOCS_VERSION is the entry
# the dropdown highlights ("vX.Y.Z" for a release, "dev" for the dev branch);
# DOCS_SWITCHER_URL is the absolute URL of the shared switcher.json at the site
# root. Local builds fall back to the pyproject version + a relative URL.
version_match = os.environ.get("DOCS_VERSION", release)
switcher_url = os.environ.get("DOCS_SWITCHER_URL", "/switcher.json")

# Its under [project.urls] "Source Code"
github_url = f"{toml_data['project']['urls']['Source Code']}"

# Link this build back to its exact source on GitHub: the release tag for a
# versioned build, or the dev branch for the dev build.
if version_match == "dev":
    source_ref_url = f"{github_url}/tree/dev"
    source_ref_label = "dev branch"
else:
    source_ref_url = f"{github_url}/releases/tag/{version_match}"
    source_ref_label = f"Release {version_match}"

print(f"Project: {project}, Author: {author}, Release: {release}, GitHub URL: {github_url}")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [        
    'autoapi.extension',  # For automatic API documentation                                                                                      
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',  # Add this extension
    # 'sphinx.ext.autosummary',  # Create neat summary tables     
    'sphinx.ext.viewcode',  # Add links to highlighted source code                                                                           
    'sphinx.ext.napoleon',  # For Google style docstrings                                                         
    'sphinx_autodoc_typehints',  # For type hints     
    'myst_parser',  # For Markdown support                                                       
]      

templates_path = ['_templates']
exclude_patterns = ['evutils/vis/reconstructor/rpg_e2vid/*']

# autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Type hints
autodoc_typehints = "both"
always_use_bars_union = True
typehints_defaults = "comma"

autodoc_member_order = 'bysource'

autoapi_type = "python"
autoapi_dirs = ["../../src/evutils"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_template_dir = "_templates/autoapi"
# autoapi_add_toctree_entry = True
suppress_warnings = ["autoapi.python_import_resolution"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True


autoapi_ignore = [
    "*/rpg_e2vid/*",
    "*/_native.py",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_book_theme'
html_theme = 'pydata_sphinx_theme'
html_logo = '_static/event_hexagon_broken.webp'
html_favicon = '_static/event_hexagon_broken.webp'
html_theme_options = {
    "icon_links": [
        {
            # Links to the release tag (or dev branch) this build came from.
            "name": source_ref_label,
            "url": source_ref_url,
            "icon": "fa-solid fa-tag",
        },
        {
            "name": "GitHub",
            "url": github_url,
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/evutils/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "logo": {
        "text": "EV-Utils",
    },
    # Built-in version-selector dropdown. Reads the shared switcher.json and
    # highlights the entry whose "version" matches version_match.
    "switcher": {
        "json_url": switcher_url,
        "version_match": version_match,
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # Don't fail the build when the switcher URL isn't reachable (e.g. local
    # builds, or the first deploy before switcher.json exists).
    "check_switcher": False,
}
html_static_path = ['_static']
# Registers the custom "pypi" glyph used by the PyPI icon_link above. Deferred
# so FontAwesome is loaded first.
html_js_files = [("pypi-icon.js", {"defer": "defer"})]

html_show_sourcelink = True

# html_theme_options = {
#     "repository_url": github_url,
#     "use_repository_button": True,
#     "use_issues_button": True,
#     "use_edit_page_button": True,
#     "repository_branch": "develop",
#     "path_to_docs": "docs",
#     "use_fullscreen_button": True,
# }