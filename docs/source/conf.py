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
copyright = '2024, Jakub Mandula'
author = 'Jakub Mandula'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [                                                                                                    
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # Create neat summary tables     
    'sphinx.ext.viewcode',  # Add links to highlighted source code                                                                           
    'sphinx.ext.napoleon',  # For Google style docstrings                                                         
    'sphinx_autodoc_typehints',  # For type hints                                                                 
]      

templates_path = ['_templates']
exclude_patterns = ['rpg_e2vid/*']

autosummary_generate = True  # Turn on sphinx.ext.autosummary


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
