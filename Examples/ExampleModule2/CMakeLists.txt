# @author  Barthélémy von Haller

O2_SETUP(NAME "ExampleModule2")

# Define the source and header files
set(SRCS
    src/Foo.cxx
    )

set(HEADERS
    include/ExampleModule2/Foo.h
    )

set(LIBRARY_NAME ExampleModule2)
set(BUCKET_NAME ExampleModule2_Bucket)
set(LINKDEF src/ExampleLinkDef.h)

# Unfortunately we have to explicitely tell cmake where to look for the includes of other modules we will require to
# generate the dictionary. This is because the list of include directories for the target (the library) is built
# after the dictionary is generated as we need the dictionary to build the library.
# TODO fix this
set(DICT_OTHER_MODULES_INCLUDE_DIRECTORIES
${CMAKE_CURRENT_SOURCE_DIR}/../ExampleModule1/include
    )

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ExampleModule2)
set(BUCKET_NAME ExampleModule2_Bucket)

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runExampleModule2
    SOURCES src/main.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)
