"""evutils: utilities for event-based data.

Event-based data processing and visualization utilities.

To get started::

    import evutils
    ...

Submodules
----------

.. toctree::
   :hidden:

   {% for submodule in visible_submodules %}
   {{ submodule.include_path }}
   {% endfor %}

.. list-table::
   :widths: 30 70

   {% for submodule in visible_submodules %}
   * - :py:mod:`{{ submodule.short_name }} <{{ submodule.id }}>`
     - {{ submodule.summary }}
   {% endfor %}


Notes
-----
Anything worth calling out up front.



"""

try:
    from ._version import version as __version__
except ImportError:
    # Default version if the _version.py is not generated
    __version__ = "0.0.0"


__all__ = [
    'augment', 
    'chunking', 
    'dataset', 
    'io', 
    'utils', 
    'vis', 
    'processing', 
    'random', 
    'types', 
    '__version__'
]

