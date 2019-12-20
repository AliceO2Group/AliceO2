<!-- doxy
\page refdocCMakeInstructions CMake Instructions

\cond DOXYGEN_IGNORE
/doxy -->

<!-- vim-markdown-toc GFM -->

- [CMake and CTest tips for AliceO2](#cmake-and-ctest-tips-for-aliceo2)
  - [CMake](#cmake)
    - [Instructions for contributors (aka developers' documentation)](#instructions-for-contributors-aka-developers-documentation)
    - [Typical CMakeLists.txt](#typical-cmakeliststxt)
    - [Examples](#examples)
      - [Ex1 Adding a basic library](#ex1-adding-a-basic-library)
      - [Ex2 Adding a basic library with a Root dictionary](#ex2-adding-a-basic-library-with-a-root-dictionary)
      - [Ex3 Adding an executable](#ex3-adding-an-executable)
      - [Ex4 Adding a couple of tests](#ex4-adding-a-couple-of-tests)
      - [Ex5 Adding a man page](#ex5-adding-a-man-page)
  - [CTest](#ctest)
    - [Selecting/excluding by test name (-R/-E)](#selectingexcluding-by-test-name--r-e)
    - [Selecting/excluding by label (-L/-LE)](#selectingexcluding-by-label--l-le)
    - [Speeding up ctest execution](#speeding-up-ctest-execution)

<!-- vim-markdown-toc -->

<!-- doxy
\endcond
/doxy -->

# CMake and CTest tips for AliceO2

## CMake

> Note that this document describe the [new CMake system](CMakeMigration.md) : the one based on buckets has been discontinued.

### Instructions for contributors (aka developers' documentation)

A sub-module `CMakeLists.txt` defines one or more _targets_.
A target generally corresponds to an actual build artifact like a (static or shared) library or an executable. Targets are the cornerstone of any modern cmake build system.

### Typical CMakeLists.txt

A typical module's `CMakeLists.txt` contains

- a call to [o2_add_library](../cmake/O2AddLibrary.cmake) to define a library (and its dependencies)
- call(s) to [o2_add_executable](../cmake/O2AddExecutable.cmake) to define one or more executables (and their dependencies)
- call(s) to [o2_add_test](../cmake/O2AddTest.cmake) to define one or more tests (and their dependencies)

Optionally it might contain a call to [o2_target_root_dictionary](../cmake/O2TargetRootDictionary.cmake) if the module's library requires a Root dictionary.

All _direct_ dependencies must be _explicitely_ defined with the `PUBLIC_LINK_LIBRARIES` keyword of the various o2_xxx functions.

Note that despite the parameter name, the `PUBLIC_LINK_LIBRARIES` should refer to _target_ names, not library names. You _have to_ use the fully qualified `O2::targetName` and not the short `basename` you might have used to _create_ the target. Note also that if the referenced target does not exist, CMake will tell you right away at the configure stage (which is a good thing).

Note also CMakeLists.txt should be considered as code and so the same care you put into writing code (e.g. do not repeat yourself, comments, etc...) should be applied to CMakeLists.txt. Also, like the rest of our code, we can take of the formatting using the [cmake-format](https://github.com/cheshirekow/cmake_format) tool (that tool is certainly not as robust as `clang-format` but it can get most of the job done easily).

### Examples

The example outputs below are from a Mac, so the shared library extension is `dylib`. On Linux it would be `so`.

#### [Ex1](../Examples/Ex1) Adding a basic library

Using the following source dir :

    Ex1
    ├── CMakeLists.txt
    ├── include
    │   └── Ex1
    │       └── A.h
    └── src
        ├── A.cxx
        ├── B.cxx
        └── B.h

With that `CMakeLists.txt` :

    o2_add_library(Ex1 SOURCES src/A.cxx src/B.cxx PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)

will define a library with 2 source files, that depends on the FairMQ::FairMQ target.

When doing a `cmake --build .` you'll find the library in the `stage/lib` dir. For instance, on a Mac :

    > ls stage/lib/*Ex1*
    stage/lib/libO2Ex1.dylib

The built library dependencies can be inspected with `otool -L` (macos) or `ldd` (linux)

    > otool -L stage/lib/libO2Ex1.dylib
    stage/lib/libO2Ex1.dylib:
            @rpath/libO2Ex1.dylib (compatibility version 0.0.0, current version 0.0.0)
            @rpath/libFairMQ.1.4.dylib (compatibility version 1.4.0, current version 1.4.2)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_container.dylib (compatibility version 0.0.0, current version 0.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.250.1)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_program_options.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_filesystem.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_system.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_regex.dylib (compatibility version 0.0.0, current version 0.0.0)
            @rpath/libFairLogger.1.2.dylib (compatibility version 1.2.0, current version 1.2.0)
            /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.4)

Where you can find the direct dependency you've specified on `FairMQ`. The rest (boost, FairLogger) are transitive dependencies (coming from FairMQ) that CMake automatically added.

And upon install `cmake --build . -- install` (the lonely `--` is not a typo) the library will be in the installation path `lib` dir and its public includes (only `A.h` in this case, not `B.h`) in the `include/Ex1` dir :

    > ls [install_topdir]
    ├── include
    │   ├── Ex1
    │       └── A.h
    ├── lib
    │   ├── libO2Ex1.dylib

#### [Ex2](../Examples/Ex2) Adding a basic library with a Root dictionary

Using a slightly modified version of the previous example (the [A.h](../Examples/Ex2/include/Ex2/A.h) now uses ClassDef for instance), we'll now add a Root dictionary :

    Ex2
    ├── CMakeLists.txt
    ├── include
    │   └── Ex2
    │       └── A.h
    └── src
        ├── A.cxx
        ├── B.cxx
        ├── B.h
        └── Ex2LinkDef.h

    o2_add_library(Ex2 SOURCES src/A.cxx src/B.cxx PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)
    o2_target_root_dictionary(Ex2 HEADERS include/Ex2/A.h src/B.h LINKDEF src/Ex2LinkDef.h)

will define a library, that depends on the `FairMQ::FairMQ` target, with 3 source files (the provided `A.cxx` and `B.cxx` plus a dictionary added by the `o2_target_root_dictionary`). In addition to create a dictionary source file, the `o2_target_root_dictionary` function also appends a dependency on `ROOT::RIO` target to `Ex2` because it's needed at link time for the dictionary part.

While the `HEADERS` parameter to `o2_target_root_dictionary` is mandatory, the `LINKDEF` one can be omitted if the LinkDef file is named \[targetBaseName]LinkDef.h and is located in the \[targetBaseName] source directory (so in the above example it could have been omitted).

When doing a `cmake --build .` you'll now find, in addition to the library in the `stage/lib` dir, a `rootmap` file and a `pcm` file. Those two files must be collocated with the library if you want to be able to load that library easily from the Root prompt.

    > ls stage/lib/*Ex2*
    stage/lib/G__O2ExDict_rdict.pcm
    stage/lib/libO2Ex2.dylib
    stage/lib/libO2Ex2.rootmap

If you look at the dependencies for libO2Ex2, you'll find the same ones as libO2Ex1, plus some ROOT ones, due to the dictionary inclusion.

    > otool -L stage/lib/libO2Ex2.dylib
    stage/lib/libO2Ex2.dylib:
            @rpath/libO2Ex2.dylib (compatibility version 0.0.0, current version 0.0.0)
            @rpath/libFairMQ.1.4.dylib (compatibility version 1.4.0, current version 1.4.2)
            @rpath/libRIO.6.16.so (compatibility version 6.16.0, current version 6.16.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_container.dylib (compatibility version 0.0.0, current version 0.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.250.1)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_program_options.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_filesystem.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_system.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_regex.dylib (compatibility version 0.0.0, current version 0.0.0)
            @rpath/libFairLogger.1.2.dylib (compatibility version 1.2.0, current version 1.2.0)
            @rpath/libThread.6.16.so (compatibility version 6.16.0, current version 6.16.0)
            @rpath/libCore.6.16.so (compatibility version 6.16.0, current version 6.16.0)
            /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.4)

The include installation will be similar to Ex1 : the `LinkDef.h` file is _not_ installed (unless you put it under include/Ex2 but that would not be wise).

    > ls [install_topdir]
    ├── include
    │   ├── Ex2
    │       └── A.h
    ├── lib
    │   ├── G__O2ExDict_rdict.pcm
    │   ├── libO2Ex2.dylib
    │   ├── libO2Ex2.rootmap

#### [Ex3](../Examples/Ex3) Adding an executable

Adding an executable to previous example :

    Ex3
    ├── CMakeLists.txt
    ├── include
    │   └── Ex3
    │       └── A.h
    └── src
        ├── A.cxx
        ├── B.cxx
        ├── B.h
        ├── Ex2LinkDef.h
        └── run.cxx

    o2_add_library(Ex3
            SOURCES src/A.cxx src/B.cxx
            PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)

    o2_target_root_dictionary(Ex3
            HEADERS include/Ex3/A.h src/B.h)

    o2_add_executable(ex3
            SOURCES src/run.cxx
            PUBLIC_LINK_LIBRARIES O2::Ex3 O2::Ex2
            COMPONENT_NAME example)

There are three things to note in the `o2_add_executable`.

First, we reference the `Ex3` library, built in the same module, using its fully qualified name `O2::Ex3`. Using just `Ex3` would not work, as there is no target named `Ex3` (Ex3 is just the basename of the target). Likewise, the "external" (to this module) Ex2 library is also referenced by its fully qualified name `O2::Ex2`. The target dependencies gather the link dependencies (to the relevant libraries used during linking) but also the include dependencies (so that the relevant include directories, e.g. `include/Ex2` are found when compiling).

Second, we used the optional `COMPONENT_NAME` argument, that will be used as part of the executable name. The output executable name will be `o2-example-ex3`, following our [naming convention for executable](https://rawgit.com/AliceO2Group/CodingGuidelines/master/naming_formatting.html#Executable_Names).

Runtime dependencies of the executable can be seen with the same `otool -L` (mac) or `ldd` (linux) command :

    > otool -L stage/bin/o2-example-ex3
    stage/bin/o2-example-ex3:
            @rpath/libO2Ex3.dylib (compatibility version 0.0.0, current version 0.0.0)
            @rpath/libO2Ex2.dylib (compatibility version 0.0.0, current version 0.0.0)
            @rpath/libFairMQ.1.4.dylib (compatibility version 1.4.0, current version 1.4.2)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_container.dylib (compatibility version 0.0.0, current version 0.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.250.1)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_program_options.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_filesystem.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_system.dylib (compatibility version 0.0.0, current version 0.0.0)
            /Users/laurent/alice/cmake/sw/osx_x86-64/boost/v1.68.0-1/lib/libboost_regex.dylib (compatibility version 0.0.0, current version 0.0.0)
            @rpath/libFairLogger.1.2.dylib (compatibility version 1.2.0, current version 1.2.0)
            @rpath/libRIO.6.16.so (compatibility version 6.16.0, current version 6.16.0)
            @rpath/libThread.6.16.so (compatibility version 6.16.0, current version 6.16.0)
            @rpath/libCore.6.16.so (compatibility version 6.16.0, current version 6.16.0)
            /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.4)

where you can see the dependencies on our two Ex2 and Ex3 libraries, as well as their dependencies (FairMQ, ROOT) and their dependencies (FairLogger).

Third, the created executable can be launched "as is" from the build tree, without having to setup the `PATH` and/or `LD_LIBRARY_PATH` environment variables.

    > stage/bin/o2-example-ex3
    Hello from ex2::A ctor
    Hello from ex3::A ctor

That is because the [RPATH](https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling) was correctly set by CMake for the build tree. It should be set correctly also when installing.

#### [Ex4](../Examples/Ex4) Adding a couple of tests

Let's add two basic test2 to our previous example.

    o2_add_test(test1
                SOURCES test/test1.cxx
                PUBLIC_LINK_LIBRARIES O2::Ex4
                COMPONENT_NAME Ex4
                LABELS fast dummy obvious
                INSTALL)

    o2_add_test(test2
                SOURCES test/test2.cxx
                PUBLIC_LINK_LIBRARIES O2::Ex4 O2::Ex3 O2::Ex2
                COMPONENT_NAME Ex4
                LABELS fast dummy)

Those two test executables will be under `stage/bin` with a name starting with `o2-test-ex4` (i.e. the COMPONENT_NAME is used but transformed into lowercase) :

    stage/
    ├── bin
    │   ├── o2-example-ex3
    │   ├── o2-example-ex4
    │   ├── o2-test-ex4-test1
    │   └── o2-test-ex4-test2
    └── lib
        ├── libO2Ex1.dylib
        ├── libO2Ex2.dylib
        ├── libO2Ex3.dylib
        └── libO2Ex4.dylib

By default tests are _not_ installed, unless the `INSTALL` option is given to `o2_add_test`. So in the installation zone only the first test will be available, under the `tests` subdirectory. So the full installation of our 4 examples would give :

    ../install-Debug/
    ├── bin
    │   ├── o2-example-ex3
    │   └── o2-example-ex4
    ├── include
    │   ├── Ex1
    │   │   └── A.h
    │   ├── Ex2
    │   │   └── A.h
    │   ├── Ex3
    │   │   └── A.h
    │   └── Ex4
    │       └── A.h
    ├── lib
    │   ├── G__O2Ex2Dict_rdict.pcm
    │   ├── G__O2Ex3Dict_rdict.pcm
    │   ├── G__O2Ex4Dict_rdict.pcm
    │   ├── libO2Ex1.dylib
    │   ├── libO2Ex2.dylib
    │   ├── libO2Ex2.rootmap
    │   ├── libO2Ex3.dylib
    │   ├── libO2Ex3.rootmap
    │   ├── libO2Ex4.dylib
    │   └── libO2Ex4.rootmap
    ├── share
    │   └── config
    │       └── rootmanager.dat
    └── tests
        └── o2-test-ex4-test1

As normally our tests are based on the [Boost.Test](https://www.boost.org/doc/libs/1_70_0/libs/test/doc/html/index.html) the dependency to the `Boost::unit_test_framework` is added automatically by the `o2_add_test` function (unless the `NO_BOOST` option is specified).

Finally, note the usage of the `LABELS` (plural) option, which can be used to categorize the tests and/or to select which tests to be ran.

    > ctest
    Test project /Users/laurent/alice/cmake/standalone/O2/build-Debug
        Start 1: O2test-ex4-test1
    1/3 Test #1: O2test-ex4-test1 ......................   Passed    0.07 sec
        Start 2: O2test-ex4-test2
    2/3 Test #2: O2test-ex4-test2 ......................   Passed    0.07 sec
        Start 3: ensure-executable-naming-convention
    3/3 Test #3: ensure-executable-naming-convention ...   Passed    0.03 sec

    100% tests passed, 0 tests failed out of 3

    Label Time Summary:
    dummy      =   0.14 sec*proc (2 tests)
    fast       =   0.14 sec*proc (2 tests)
    obvious    =   0.07 sec*proc (1 test)

    > ctest -L obvious # run only tests with a label of "obvious"
    Test project /Users/laurent/alice/cmake/standalone/O2/build-Debug
        Start 1: O2test-ex4-test1
    1/3 Test #1: O2test-ex4-test1 ......................   Passed    0.07 sec

    100% tests passed, 0 tests failed out of 1

    Label Time Summary:
    dummy      =   0.07 sec*proc (1 test)
    fast       =   0.07 sec*proc (1 test)
    obvious    =   0.07 sec*proc (1 test)

Note that tests can also be selected by name using regexp (`-R`).
Tests can also be _excluded_ based on label (`-LE`) or name (`-RE`).

    Test project /Users/laurent/alice/cmake/standalone/O2/build-Debug
        Start 2: O2test-ex4-test2
    1/2 Test #2: O2test-ex4-test2 ......................   Passed    0.07 sec
        Start 3: ensure-executable-naming-convention
    2/2 Test #3: ensure-executable-naming-convention ...   Passed    0.03 sec

    100% tests passed, 0 tests failed out of 2

    Label Time Summary:
    dummy    =   0.07 sec*proc (1 test)
    fast     =   0.07 sec*proc (1 test)

#### [Ex5](../Examples/Ex5) Adding a man page

If a module provides one or more executables, it might be of interest for the users of those executables to have access to a man page for them. Ex5 illustates that use case.

    .
    ├── CMakeLists.txt
    ├── README.md
    ├── doc
    │   └── ex5.7.in
    └── src
        └── run.cxx

The [man page](ManPages.md) is created using :

    o2_target_man_page([targetName] NAME ex5 SECTION 7)

where `NAME xx` refers to a file `doc/xx.[SECTION].in`, and the actual `targetName` can be found from the base target name (ex5 in that case) using the [o2_name_target](../cmake/O2NameTarget.cmake) function.

## CTest

In the build directory of O2, if you launch the `ctest` command, all the O2 tests will be ran, which is not always what you want/need, in particular during development.

As shown above in [Ex4](#ex4-adding-a-couple-of-tests), there are fortunately various ways to narrow the set of tests which are ran by `ctest`.

Note that in all the commands below, the `-N` option is used to show the list of test that would be selected, without running them. Remove that option to actually run the tests.

### Selecting/excluding by test name (-R/-E)

Probably the simplest is to select the test name, using the `-R` (regexp) option :

```shell
> ctest -N -R test/Contour
  Test #269: Detectors/MUON/MCH/Contour/test/Contour.cxx
  Test #270: Detectors/MUON/MCH/Contour/test/ContourCreator.cxx

>  ctest -N -R "test/Contour\.(.)"
  Test #269: Detectors/MUON/MCH/Contour/test/Contour.cxx

```

Instead of selecting the tests to run, you can invert the logic and exclude tests matching a regular expression using `-E` instead of `-R`.

### Selecting/excluding by label (-L/-LE)

Assuming the tests have been labelled, you can use `-L` (to include) or `-LE` (to exclude) tests based on their labels. The full list of defined labels can be shown using the `--print-labels` option.

```shell
> ctest -N -L mch
  Test #268: Detectors/MUON/MCH/Contour/test/BBox.cxx
  Test #269: Detectors/MUON/MCH/Contour/test/Contour.cxx
  Test #270: Detectors/MUON/MCH/Contour/test/ContourCreator.cxx
  Test #271: Detectors/MUON/MCH/Contour/test/Edge.cxx
  Test #272: Detectors/MUON/MCH/Contour/test/Interval.cxx
  Test #273: Detectors/MUON/MCH/Contour/test/Polygon.cxx
  Test #274: Detectors/MUON/MCH/Contour/test/SegmentTree.cxx
  Test #275: Detectors/MUON/MCH/Contour/test/Vertex.cxx
  Test #276: Detectors/MUON/MCH/Simulation/macros/drawMCHGeometry.C
  Test #277: Detectors/MUON/MCH/Simulation/macros/drawMCHGeometry.C_compiled
```

### Speeding up ctest execution

Finally, note that `ctest` can run tests in parallel (unless those that are explicitely marked as not safe to run in parallel) using the `-j n` option, where `n` is the max number of tests to run in parallel.

Serial execution :

```shell
> ctest -L mch -LE long -E Raw
    Start 265: Detectors/MUON/MCH/Contour/test/BBox.cxx
1/8 Test #265: Detectors/MUON/MCH/Contour/test/BBox.cxx .............   Passed    0.04 sec
    Start 266: Detectors/MUON/MCH/Contour/test/Contour.cxx
2/8 Test #266: Detectors/MUON/MCH/Contour/test/Contour.cxx ..........   Passed    0.05 sec
    Start 267: Detectors/MUON/MCH/Contour/test/ContourCreator.cxx
3/8 Test #267: Detectors/MUON/MCH/Contour/test/ContourCreator.cxx ...   Passed    0.05 sec
    Start 268: Detectors/MUON/MCH/Contour/test/Edge.cxx
4/8 Test #268: Detectors/MUON/MCH/Contour/test/Edge.cxx .............   Passed    0.05 sec
    Start 269: Detectors/MUON/MCH/Contour/test/Interval.cxx
5/8 Test #269: Detectors/MUON/MCH/Contour/test/Interval.cxx .........   Passed    0.05 sec
    Start 270: Detectors/MUON/MCH/Contour/test/Polygon.cxx
6/8 Test #270: Detectors/MUON/MCH/Contour/test/Polygon.cxx ..........   Passed    0.05 sec
    Start 271: Detectors/MUON/MCH/Contour/test/SegmentTree.cxx
7/8 Test #271: Detectors/MUON/MCH/Contour/test/SegmentTree.cxx ......   Passed    0.05 sec
    Start 272: Detectors/MUON/MCH/Contour/test/Vertex.cxx
8/8 Test #272: Detectors/MUON/MCH/Contour/test/Vertex.cxx ...........   Passed    0.05 sec

100% tests passed, 0 tests failed out of 8

Label Time Summary:
mch     =   0.38 sec*proc (8 tests)
muon    =   0.38 sec*proc (8 tests)

Total Test time (real) =   0.49 sec

```

Parallel execution (on a machine with 16 cores) :

```shell
> ctest -L mch -LE long -E Raw -j 16
    Start 270: Detectors/MUON/MCH/Contour/test/Polygon.cxx
    Start 271: Detectors/MUON/MCH/Contour/test/SegmentTree.cxx
    Start 268: Detectors/MUON/MCH/Contour/test/Edge.cxx
    Start 267: Detectors/MUON/MCH/Contour/test/ContourCreator.cxx
    Start 269: Detectors/MUON/MCH/Contour/test/Interval.cxx
    Start 266: Detectors/MUON/MCH/Contour/test/Contour.cxx
    Start 265: Detectors/MUON/MCH/Contour/test/BBox.cxx
    Start 272: Detectors/MUON/MCH/Contour/test/Vertex.cxx
1/8 Test #270: Detectors/MUON/MCH/Contour/test/Polygon.cxx ..........   Passed    0.08 sec
2/8 Test #271: Detectors/MUON/MCH/Contour/test/SegmentTree.cxx ......   Passed    0.08 sec
3/8 Test #268: Detectors/MUON/MCH/Contour/test/Edge.cxx .............   Passed    0.08 sec
4/8 Test #269: Detectors/MUON/MCH/Contour/test/Interval.cxx .........   Passed    0.08 sec
5/8 Test #267: Detectors/MUON/MCH/Contour/test/ContourCreator.cxx ...   Passed    0.08 sec
6/8 Test #266: Detectors/MUON/MCH/Contour/test/Contour.cxx ..........   Passed    0.08 sec
7/8 Test #265: Detectors/MUON/MCH/Contour/test/BBox.cxx .............   Passed    0.08 sec
8/8 Test #272: Detectors/MUON/MCH/Contour/test/Vertex.cxx ...........   Passed    0.08 sec

100% tests passed, 0 tests failed out of 8

Label Time Summary:
mch     =   0.65 sec*proc (8 tests)
muon    =   0.65 sec*proc (8 tests)

Total Test time (real) =   0.19 sec

```
