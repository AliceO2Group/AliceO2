Code organisation and build
=

## Principles
* A _module_ is a set of code closely related sharing an interface that can result in one or more libraries.
* Favour is given to extracting large common components(modules/projects) into their own repositories within
  AliceO2Group in github.
* AliceO2 therefore becomes a thinner repo containing :
  * Detector specific code (e.g. related to reconstruction, simulation, calibration or qc).
  * Commonalities (e.g. DataFormat, Steer-like), i.e. things other components depend on and that have not been extracted to their own repo.
  * Global algorithms (e.g. global tracking), i.e. things that depend on several detectors.
* The directory structure can be either per detector or per function or a mixture.
  The AliceO2 repository has a mixture of _per detector_ and _per function_ sub-modules with corresponding sub-structure.
* Dependencies are defined centrally as _buckets_.
* Each sub-module generates a single library linked against the dependencies defined in a single bucket.
* sub-modules' executable(s) link against the same bucket as the library and the library itself.
* Horizontal dependencies are in general forbidden (between sub-modules at the same level) (?)
* Naming : camel-case
  * What is repeated / structural starts with a lower case letter (e.g. src, include, test).
  * The rest (labels, unique names) start with an upper case letter (e.g. Common, Detectors).

## Repository organisation
The _per detector_ sub-modules are grouped under the Detectors directory. The _per function_ are in 
the top dir or grouped, e.g. Utilites. 

Each sub-module has in principle a number of directories: 
src, include/<name of submodule>, doc, test and cmake. Depending on the case, some can be 
voluntareely left out. 

The headers go to the include directory if they are part of the interface, in the src otherwise.

## CMake instructions

This section explains the CMake functions visible to the end user. 

A sub-module CMakeLists.txt minimally contains (see Examples/ExampleModule1)
 
* The setup the system : `O2_SETUP(NAME My_Module)`
* The list of the source files in the variable SRC : `set(SRCS something.cxx)`
* The name of the library : `set(LIBRARY_NAME My_Module)`
* The name of the dependency bucket to use : `set(BUCKET_NAME My_bucket)`
* The call to generate the library : `O2_GENERATE_LIBRARY()`

Optionally it contains (see Examples/ExampleModule2)

* To generate a dictionary :
  * The list of source files not to be used for the dictionary : `set(NO_DICT_SRCS src/Bar.cxx)`
  * The list of headers (private and public) : `set(HEADERS include/${MODULE_NAME}/Foo.h)`
  * The linkdef : `set(LINKDEF src/ExampleLinkDef.h)`
* To generate an executable : 
  * Again the name of the library and bucket_name because they got erased by `O2_GENERATE_LIBRARY()`
    ```
    set(LIBRARY_NAME ExampleModule2)
    set(BUCKET_NAME ExampleModule2_bucket)
    ```
  * The call to generate the executable : 
    ```
    O2_GENERATE_EXECUTABLE(
        EXE_NAME runExampleModule1
        SOURCES src/main.cxx
        MODULE_LIBRARY_NAME ${LIBRARY_NAME}
        BUCKET_NAME ${BUCKET_NAME}
    )
    ```
    
If a new bucket is needed, it should be defined in cmake/O2Dependencies.cmake using this function : 
```
o2_define_bucket(
    NAME
    ExampleModule2_bucket

    DEPENDENCIES
    ${Boost_PROGRAM_OPTIONS_LIBRARY}    # a library
    ExampleModule1                      # another module
    ExampleModule1_bucket               # another bucket

    INCLUDE_DIRECTORIES
    ${Boost_INCLUDE_DIR}                # another lib include dir
    ${CMAKE_SOURCE_DIR}/Examples/ExampleModule1/include # another module's include dir
)
```

If it needs a new external library, it should be first discussed with CWG13.

## Developers documentation 

* Q: Why are the libraries' directories globally set in O2Dependencies.cmake ? 
 * A: CMake discourages the use of _link_directories_ because find_package and find_library
   should return absolute paths. As a consequence little effort is put in the development of this 
   feature and it only exists at the global level. We can't set it on a target like the 
   _include_directories_ for example.