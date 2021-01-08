<!-- doxy
\page refdocDoxygenInstructions Doxygen Instructions
/doxy -->

# Doxygen 

## Instructions for the contributors

The Doxygen documentation pages are generated from the `README.md` files placed in the O2 directories. 

The references between the Doxygen pages and subpages are achieved using a Doxygen `page` and `subpage` keywords. All `README.md` files must contain the Doxygen `page` tag on the top, and if the module contains other documentation pages (usually in its sub-directories), it must be linked using the Doxygen `subpage` tag. Special markdown comments are used to disable rendering of Doxygen keywords on GitHub.

*When adding a new documentation page you must always add a `\subpage` declaration in an upper category page otherwise your page will pollute the "global" references in the left tab menu.*

See the examples of `README.md` files both at the top and a submodule levels below, or follow <a href="https://github.com/AliceO2Group/AliceO2/blob/dev/doc/DoxygenInstructions.md"> this link </a> if you are in Doxygen documentation.

<!--  /// \cond EXCLUDE_FOR_DOXYGEN -->

#### An example of a `README.md` file in the O2 directories at the top level.

```
<!-- doxy
\page refModulename Module 'Modulename'
/doxy -->

# The module title

The paragraph(s) with the module description.

<!-- doxy
This module contains the following submodules:

* \subpage refModulenameSubmodulename1
* \subpage refModulenameSubmodulename2
* \subpage refModulenameSubmodulename3
/doxy -->
```
#### An example of a `README.md` file at a submodule level:

```
<!-- doxy
\page refModulenameSubmodulename1 Submodulename1
/doxy -->

# The submodule1 title

The paragraph(s) with the submodule description.
```
<!--  /// \endcond  -->

The `Modulename` and `Submodulename` in the `page` and `subpage` tags must be identical (including case match) with the directories name. 
