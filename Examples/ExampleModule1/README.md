This is an example of a basic module using the O2 CMake macros to generate a library and an executable.

In particular :
- O2_SETUP : necessary to register the module in O2.
- O2_GENERATE_LIBRARY : use once and only once per module to generate a library.
- O2_GENERATE_EXECUTABLE : generate executables.