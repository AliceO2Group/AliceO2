\page refdocDoxygenInstructions Doxygen Instructions

Doxygen
=======

## Instructions for the contributors

The Doxygen documentation pages are generated from the `README.md` files placed in the O2 directories. 

The references between the Doxygen pages and subpages are achieved using a Doxygen `page` and `subpage` keywords. All `README.md` files must contain the Doxygen `page` tag on the top, and if the module contains other documentation pages (usually in its sub-directories), it must be linked using the Doxygen `subpage` tag.

*When adding a new documentation page you must always add a `\subpage` declaration in an upper category page otherwise your page will pollute the "global" references in the left tab menu.*

#### An example of a `README` file in the O2 directories at the top level.

    \page refModulename Module 'Modulename'

    The module title
    ================

    The paragraph(s) with the module description.

    This module contains the following submodules:
    
    - \subpage refModulenameSubmodulename1
    - \subpage refModulenameSubmodulename2
    - \subpage refModulenameSubmodulename3


#### An example of a `README` file at a submodule level:

    \page refModulenameSubmodulename1 Submodulename1

    The submodule1 title
    ====================

    The paragraph(s) with the submodule description.

The `Modulename` and `Submodulename` in the `page` and `subpage` tags must be identical (including case match) with the directories name. 
