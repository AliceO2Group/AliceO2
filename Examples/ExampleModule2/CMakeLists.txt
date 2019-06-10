# Example of a CMakeLists.txt for a module that requires a Root dictionary and
# depends on the ExampleModule1 library
#
# We specify the regular source files for the library, no mention of Root
# dictionary specifics here.
#
# We are explicit about the fact that we depend on O2::ExampleModule1
#
o2_add_library(ExampleModule2
               SOURCES src/Foo.cxx src/Bar.cxx
               PUBLIC_LINK_LIBRARIES O2::ExampleModule1)

# The o2_target_root_dictionary function will generate a dictionary source file
# and append it to the list of sources of the target ExampleModule2
#
# The HEADERS argument is mandatory and must contain the list of include files
# needed to generate the dictionary.
#
# The LINKDEF parameter is only required if the LinkDef file is not named
# src/[targetName]LinkDef.h
#
o2_target_root_dictionary(ExampleModule2
                          HEADERS src/Bar.h include/ExampleModule2/Foo.h
                          LINKDEF src/ExampleLinkDef.h)

o2_add_executable(runExampleModule2
                  SOURCES src/main.cxx
                  COMPONENT_NAME example
                  PUBLIC_LINK_LIBRARIES O2::ExampleModule2
                                        Boost::program_options)
