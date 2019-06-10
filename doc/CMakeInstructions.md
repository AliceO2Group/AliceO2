\page refdocCMakeInstructions CMake Instructions

# CMake

## Instructions for contributors (aka developers' documentation)

A sub-module `CMakeLists.txt` defines one or more _targets_.
A target generally corresponds to an actual build artifact like a (static or shared) library or an executable. Targets are the cornerstone of any modern cmake build system.

Targets are normally created with CMake-provided functions like `add_library`, `add_executable` and characterized with functions like `target_include_directories`, `target_link_libraries`, etc...

We pondered for a long time whether we should simply stick to those native functions for our builds. But that would mean quite a bit of repetitive pieces of code in all our CMakeLists.txt. Plus it would make enforcing some conventions harder. So we decided (like in the previous incarnation of our build system) to use our own functions instead.

Compared to the previous system though, we tried :

- to use names (of the functions and their parameters) closely matching those of the original CMake ones, so people already CMake are less confused
- to use only functions instead of macros (unless required), so the parameters do not leak into parent scope
- to forego completely the usage of variables (variables are not bad practice per se, but most of our CMakeLists.txt can be written without any)

As a side note, CMakeLists.txt should be considered as code and so the same care you put into writing code (e.g. do not repeat yourself, comments, etc...) should be applied to CMakeLists.txt. Also, like the rest of our code, we can take of the formatting using the [cmake-format](https://github.com/cheshirekow/cmake_format) tool (that tool is certainly not as robust as `clang-format` but it can get most of the job done easily).

All our CMake functions are defined in the [cmake](../cmake) directory. Each file there defines one function. The filename is [UpperCamelCase](https://en.wikipedia.org/wiki/Camel_case) while the function name is [snake_case](https://en.wikipedia.org/wiki/Snake_case) (CMake function names are case insensitive but the modern convention is to have them all lower case). So for instance `o2_add_executable` is defined in `cmake/O2AddExecutable.cmake`. Each function is documented in its corresponding `.cmake` file.

## Typical CMakeLists.txt

A typical module's `CMakeLists.txt` contains (see [Examples/ExampleModule1](../Examples/ExampleModule1)

- a call to `o2_add_library` to define a library (and its dependencies)
- call(s) to `o2_add_executable` to define one or more executables (and their dependencies)
- call(s) to `o2_add_test` to define one or more tests (and their dependencies)

Optionally it might contain a call to `o2_target_root_dictionary` (see [Examples/ExampleModule2](../Examples/ExampleModule2/) if the module's library requires a Root dictionary.

Compared to the previous system, there is no notion of bucket. All dependencies are explicitely defined with the `PUBLIC_LINK_LIBRARIES` (or less commonly with `PRIVATE_LINK_LIBRARIES`) keyword of the various o2_xxx functions.

Note that despite the parameter name, the `PUBLIC_LINK_LIBRARIES` should refer to _target_ names, not library names. You _have to_ use the fully qualified `O2::targetName` and not the short `basename` you might have used to _create_ the target. Note also that if the referenced target does not exist, CMake will tell you right at the configure stage (which is a good thing).
