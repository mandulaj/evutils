# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os                                                                                                         
import sys                                                                                                        
sys.path.insert(0, os.path.abspath('../../src'))  


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'evutils'
copyright = '2026, Jakub Mandula'
author = 'Jakub Mandula'
release = 'v0.3.0'
github_url = "https://github.com/mandulaj/evutils"

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
            "name": "GitHub",
            "url": github_url,
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "EV-utils",
    },

}
# html_static_path = ['_static']

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