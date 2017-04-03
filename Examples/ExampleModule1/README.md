Example 1
=========

[TOC]

# Introduction {#Introduction}

This is an example of a basic module using the O2 CMake macros to generate a library and an executable.

## Macros and variables {#Macros}

- O2_SETUP : necessary to register the module in O2.
- O2_GENERATE_LIBRARY : use once and only once per module to generate a library.
- O2_GENERATE_EXECUTABLE : generate executables.
- HEADERS is not needed in case we don't generate a dictionary.

## Classes {#Classes}

We have one class that belongs to the interface (Foo) and one class that is internal (Bar).
As a consequence, the header of the former should go to the include directory whereas the header
of the second must go to the src directory.

Foo uses Bar, both are included in the library. The executable uses Foo.

# Documentation {#Documentation}

The documentation is in markdown with special markers for doxygen such as `[TOC]`.
Note the that the TOC will work only if no levels are skip (don't create a subsection without a section
above it).
