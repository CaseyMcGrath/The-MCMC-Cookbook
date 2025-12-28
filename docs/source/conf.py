# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'The MCMC Cookbook'
copyright = '2025, Casey McGrath'
author = 'Casey McGrath'



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# !!EXTENSIONS Note!!
# --> myst_nb supercedes myst_parser when it is used, and myst_parser should be commented out!
extensions = [#"myst_parser",      # https://myst-parser.readthedocs.io/en/latest/intro.html
"sphinxcontrib.bibtex",           # https://sphinxcontrib-bibtex.readthedocs.io/en/latest/quickstart.html
"sphinx_design",                  # https://myst-parser.readthedocs.io/en/latest/intro.html#extending-sphinx
"myst_nb",                        # https://myst-nb.readthedocs.io/en/latest/quickstart.html 
"sphinx_copybutton",              # https://sphinx-copybutton.readthedocs.io/en/latest/
]

templates_path = ['_templates']
exclude_patterns = []

bibtex_bibfiles = ['references.bib']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = "The MCMC Cookbook"


html_theme_options = {
    "repository_url": "https://github.com/CaseyMcGrath/The-MCMC-Cookbook",
    "use_repository_button": True,
}


