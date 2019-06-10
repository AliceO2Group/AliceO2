# Example of a minimal CMakeLists.txt which creates :
#
# * one library
# * one executable
# * one test

# Define a target of type "library" with a basename of ExampleModule1
#
# As per our code organisation policy, all the includes that are in
# include/ExampleModule1 are considered public and will be installed, even if we
# don't explicitely specify them in the o2_add_library call
#
# As per our naming conventions, the built library will be named
# libO2ExampleModule1.(dylib|so|a).
#
# The cmake target itself will have a name derived from ExampleModule1 (see the
# details in o2_name_target function if you really are interested), so you'll
# usually want to reference it afterwards by its alias O2::ExampleModule1 in
# order not to depend on how the final target name is computed.
#
o2_add_library(ExampleModule1 SOURCES src/Foo.cxx src/Bar.cxx)

# Define a target of type "executable" that depends on the library above.
#
# Following our naming conventions, the built executable will be named
# o2-example-module1
#
o2_add_executable(module1
                  SOURCES src/main.cxx
                  COMPONENT_NAME example
                  PUBLIC_LINK_LIBRARIES O2::ExampleModule1
                                        Boost::program_options)

# Define a test executable that depends on the library above.
#
# Following our naming conventions, the built executable will be named
# o2-test-example-module1
# 
# The labels attached to the test might be used to narrow the ctest
# execution to specific tests:
#
# ctest -L fast 
# ctest -L example
#
o2_add_test(module1
            SOURCES test/testExampleModule1.cxx
            COMPONENT_NAME example
            LABELS fast example
            PUBLIC_LINK_LIBRARIES O2::ExampleModule1)
