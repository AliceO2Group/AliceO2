CMake
=====

## Instructions for the contributors

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
    ${CMAKE_SOURCE_DIR}/Examples/ExampleModule1/include   # another module's include dir

    SYSTEMINCLUDE_DIRECTORIES
    ${Boost_INCLUDE_DIR}                # a system lib include dir
)
```

If it needs a new external library, it should be first discussed with WP3.

## Developers' documentation

* Q: Why are the libraries' directories globally set in O2Dependencies.cmake ?
 * A: CMake discourages the use of _link_directories_ because find_package and find_library
   should return absolute paths. As a consequence little effort is put in the development of this
   feature and it only exists at the global level. We can't set it on a target like the
   _include_directories_ for example.
* Q: Why buckets ?
 * A: The goal is to avoid a dependency nightmare.
 It allows to define centrally and in an organized way the dependencies for all modules.
 It also allows us to be especially careful in PRs about changes to the bucket definition file.
* Q: Why macros to build libraries and executables ?
 * A: To simplify the life of the users and to make sure everyone does it the same way. It is also a way
 to reuse what was made for FairRoot.
