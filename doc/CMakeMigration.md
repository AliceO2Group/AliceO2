# Migration to "Modern" CMake

## Big picture

The driving force is to abandon the whole bucket system that proved itself a great way to hide genuine dependency issues we're having in our build system.

To do so the main idea is to migrate our CMake usage to latest practices recommended by the CMake community.

Details can be found in many online locations in [blog post](https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/), [video](https://www.youtube.com/watch?v=bsXLMQ6WgIk), or [book](https://crascit.com/professional-cmake/) form, but the core concept is to base everything on **targets** and forego as much as possible the usage of variables and/or directory specific functions.

Also, since our CMakeLists.txt that were first written a few years back, more third-party libraries have embraced the new CMake ways of working and as such provide reasonably good `XXXConfig.cmake` files that are used when using `find_package(XXX)`. We should use them instead of cooking complicated `FindXXX.cmake` as much as we can.

Targets are normally created with CMake-provided functions like `add_library`, `add_executable` and characterized with functions like `target_include_directories`, `target_link_libraries`, etc...

We pondered for a long time whether we should simply stick to those native functions for our builds. But that would mean quite a bit of repetitive pieces of code in all our CMakeLists.txt. Plus it would make enforcing some conventions harder. So we decided (like in the previous incarnation of our build system) to use our own functions instead.

Compared to the previous system though, we tried :

- to use names (of the functions and their parameters) closely matching those of the original CMake ones, so people that already know CMake are less confused
- to use only functions instead of macros (unless required), so the parameters do not leak into parent scope
- to forego (almost) completely the usage of custom variables (variables are not bad practice per se, but most of our CMakeLists.txt can be written without any)

## Custom CMake functions

All our CMake functions are defined in the [cmake](../cmake) directory. Each file there defines one function. The filename is [UpperCamelCase](https://en.wikipedia.org/wiki/Camel_case) while the function name is [snake_case](https://en.wikipedia.org/wiki/Snake_case) (CMake function names are case insensitive but the modern convention is to have them all lower case). So for instance `o2_add_executable` is defined in `cmake/O2AddExecutable.cmake`. Each function is documented in its corresponding `.cmake` file.

The main defined functions are currently :

- [o2_add_executable](../cmake/O2AddExecutable.cmake)
- [o2_add_library](../cmake/O2AddLibrary.cmake)
- [o2_add_header_only_library](../cmake/O2AddHeaderOnleLibrary.cmake)
- [o2_add_test](../cmake/O2AddTest.cmake)
- [o2_add_test_wrapper](../cmake/O2AddTestWrapper.cmake)
- [o2_target_root_dictionary](../cmake/O2TargetRootDictionary.cmake)

All the `o2_` functions above take as first (unnamed) parameter the _basename_ of a target. In order to prepare for the packaging step, the actual target name is _not_ the same as this _basename_. The target naming scheme is handled by the `o2_name_target` function. But in most cases the developpers do not need to know this final name, just that their target should be referenced as `basename` when they are used as first parameter of the `o2_xxx` functions or as `O2::basename` when used as dependencies.

## Migration plan

The idea is to go in steps.

1. rewrite all our main CMakeLists.txt to get a working build, but without taking care of the more difficult or less critical parts, like a) GPU stuff (critical and difficult) b) testing of Root macros (difficult) c) getting a proper O2Config.cmake produced (aka packaging). This first step is like a proof-of-concept, but almost full scale. Discuss the implementation choices at this stage.

2. add GPU/HIP/OpenCL stuff

3. add creation of O2Config.cmake

4. add Root macro testing

5. polish and fix the remaining inconsistencies (e.g. test labelling, etc...)

## Step 1

In this step the idea is to limit ourselves to change only `CMakeLists.txt` and `*.cmake` files and not the source code (unless absolutely necessary). The top [CMakeLists.txt](../CMakeLists.txt) was rewritten from scratch and is composed of different parts :

- preamble: basic project definition, cmake version requirement, ctest inclusion
- project wide setup: cxx checks, build options, output paths, rpath settings
- external dependencices: all the find_package calls
- definition of the targets : i.e. inclusion of all the sub_directories, in the correct order
- end with testing and doc

For the developper there is one major visible usage change : testing (still using ctest of course) can now be done from the build tree itself, i.e. without installation.

The main changes with respect to the current/previous situation are highlighted below.

### Preamble

CMake 3.13 is still the minimum version required. 3.14 would be interesting but is not critical. Note that when out CMake 3.15 will bring some generator expressions features that we might want to take advantage of (e.g. [REMOVE_DUPLICATES](https://gitlab.kitware.com/cmake/cmake/issues/18210)) and so might justify bumping the CMake version we use at that moment.

### Project wide setup

Here we have some basic sanity checks (forbid in-source builds for instance), perform some feature checks on the CXX compiler, set the default for our [build options](../cmake/O2DefineOptions.cmake), set the output directories and RPATH settings. Note the `BUILD_SIMULATION` option : currently used "only" to fetch more dependencies, but might imagine to actually group all the simulation-dependent parts of O2 into an optional component ? Not for now, but maybe a thing to consider for the future ?

Compared to previous usage, a `stage` area was added in the build tree, where (most of) the build artifacts are created. That's where you'll find the `bin`, `lib`, `share` directories instead of directly under the build topdir.

### External dependencies

Third-party dependencies are found in [dependencies/CMakeLists.txt](../dependencies/CMakeLists.txt).
`find_package` calls are of the `CONFIG` variety unless there's a compelling reason to use the `MODULE` version.
In particular :

- [FindFairRoot](../dependencies/FindFairRoot.cmake) is creating imported targets so we can use `FairRoot::XXX` targets even if those are not (yet) created by FairRoot itself. That file will disappear when FairRoot completes its own CMake migration.
- [FindGeant3](../dependencies/FindGeant3.cmake), [FindGeant4](../dependencies/FindGeant4.cmake), [Geant4VMC](../dependencies/FindGeant4VMC.cmake) : while those MC projects have Config.cmake files and thus define targets, they do not include the proper include paths for those targets. So those Find modules are just light ones that add the missing include paths to the imported targets defined in those projects.
- [Findpythia](../dependencies/Findpythia.cmake) (for Pythia8) and [Findpythia6](../dependencies/Findpythia6.cmake) define imported targets for those libraries
- [Findms_gsl](../dependencies/Findms_gsl.cmake) defines a `ms_gsl` target corresponding to the header only library
- [FindRapidJSON](../dependencies/FindRapidJSON.cmake) defines a `RapidJSON::RapidJSON` target corresponding to the header only library

As an aside, in order to ease the development of this new CMake system, a [o2_find_dependencies_from_alibuild](../dependencies/O2FindDependenciesFromAliBuild.cmake) function that "detects" the third-party dependencies from an AliBuild installation zone was developped. That might come in handy for other people too (e.g. those using IDEs like CLion ?)

### Definition of all targets

That part is the meat of the CMakeLists.txt and is just the inclusion of the relevant subdirectories, but with a catch : order matters ! Some dependency cycles that had been hidden with the bucket system now shows up ...

### Testing

Lastly some setup for testing is done in `tests` subdirectory. That part is still a bit WIP, but that's the location of the shell scripts that are used.

## Status at end of step 1

The way it was developped : first make a regular install of O2@dev using aliBuild.
Then switch to `cmake-migration-step-1` branch. Create a build directory somewhere, and run cmake there :

```
> cd build-RelWithDebInfo
> rm -rf *
> cmake $HOME/alice/cmake/O2 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=../install-RelWithDebInfo -DCMAKE_GENERATOR=Ninja -DALIBUILD_BASEDIR=$HOME/alice/cmake/sw/osx_x86-64
```

Use `ninja` to build, `cmake .` to force rerunning cmake. Rince and repeat. Do it on Mac(10.14) and on CentOS7.

The tests for this step (the failing test is only failing on CentOS7 in Debug configuration) :

```
~/alice/cmake/standalone/O2/build-Debug$ ctest --progress -j32
Test project /home/aphecetche/alice/cmake/standalone/O2/build-Debug
50/85 Test #34: O2test-dplutils-RootTreeWriterWorkflow..............***Failed    2.07 sec
85/85 Test #24: O2test-detectorsbase-MatBudLUT
99% tests passed, 1 tests failed out of 85

Label Time Summary:
dummy      =   0.11 sec*proc (2 tests)
example    =   0.05 sec*proc (1 test)
fast       =   0.16 sec*proc (3 tests)
gpu        =   2.13 sec*proc (2 tests)
its        =   1.06 sec*proc (1 test)
mch        =   0.42 sec*proc (8 tests)
mft        =   1.06 sec*proc (1 test)
mid        =  20.33 sec*proc (5 tests)
obvious    =   0.06 sec*proc (1 test)
slow       =  14.10 sec*proc (1 test)
steer      =   3.10 sec*proc (2 tests)
tpc        =  28.64 sec*proc (10 tests)

Total Test time (real) =  23.29 sec

The following tests FAILED:
         34 - O2test-dplutils-RootTreeWriterWorkflow (Failed)
Errors while running CTest
```

## Tips

### Getting the list of targets

In the build directory, if you do :

```
mkdir -p .cmake/api/v1/query/
touch .cmake/api/v1/query/codemodel-v2
```

Then after the cmake configure stage (if using CMake >= 3.14) you'll get a list of JSON files describing the targets in the reply subdir :

```
> tree .cmake/api/v1/reply
├── codemodel-v2-83681dd2d17b5fde868d.json
├── index-2019-06-11T10-25-54-0072.json
├── target-O2bench-mch-segmentation3-Debug-9c692e4e44d81a3a3a92.json
├── target-O2bench-mid-clusterizer-Debug-1340ba249c9a02e67ed5.json
├── target-O2bench-mid-tracker-Debug-3e2229d129fb77d3b863.json
├── target-O2exe-alicehlt-eventsampler-device-Debug-3bd0b54283972d791f58.json
├── target-O2exe-alicehlt-runcomponent-Debug-80eb80e9623ee42d3fb2.json
├── target-O2exe-alicehlt-wrapper-device-Debug-10621b10deb5caf32c18.json
├── target-O2exe-ccdb-conditions-client-Debug-07e6fca14004fe227979.json
├── target-O2exe-ccdb-conditions-server-Debug-ab5afdf6854abca507b7.json
├── target-O2exe-ccdb-standalone-client-Debug-aaff2add767f9b354708.json

```

## Step 2

This step is trying to get the GPU targets back in business. There are three versionsto consider: [HIP](../GPU/GPUTracking/Base/hip), [OpenCL](../GPU/GPUTracking/Base/opencl) and CUDA (for [TPC](..GPU/GPUTracking/Base/cuda) and [ITS](../Detectors/ITSMFT/ITS/tracking/cuda)).

In the `GPUTracking` all the AliRoot-specific references has been removed. They will need to be put back there if needed.

This step was tested on a CentOS7 server with OpenCL, HIP and CUDA dev. kits installed.

This step also brings a temporary [O2RecipeAdapter](../dependencies/O2RecipeAdapter.cmake) cmake include to be able to test this without having to modify (too much at least) the existing o2 recipe and CI.

## Step 3

In order to have the O2Suite building fine, step 3 is now the addition of the creation of a proper O2Config.cmake file, so that consumer packages (like QualityControl) can use our targets.

## Step 3 and 4

The proper generation of the O2Config.cmake (step 3) file has been done in anticipation for its usage by QualityControl.

The macro testing (step 4) is working, but only within the correct environment (i.e. within an alibuild build). The running of ctest without a prior environment will have to be deferred for later on, as it's not a completely trivial task (and is a departure from the current practice anyway)

[ ] Remaining to be done : generation of O2ConfigVersion.cmake file.
