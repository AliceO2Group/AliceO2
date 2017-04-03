Example 2
=========

[TOC]

TODO : explain what is specific to this example

This is an example of a basic module using the O2 CMake macros to generate a library and an executable.
It also generates the ROOT dictionary.

In particular :
- O2_SETUP : necessary to register the module in O2.
- O2_GENERATE_LIBRARY : use once and only once per module to generate a library.
- O2_GENERATE_EXECUTABLE : generate executables.

We have one class that belongs to the interface (Foo) and for which a dictionary is
generated and one class that is internal (Bar) without dictionary. We use NO_DICT_SRCS for the latter.
The header of Foo should go to the include directory whereas the header
of Bar must go to the src directory.

Foo uses Bar, both are included in the library. The executable uses Foo.
