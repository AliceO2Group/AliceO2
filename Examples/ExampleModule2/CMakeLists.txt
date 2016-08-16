# @author  Barthélémy von Haller

set(MODULE_NAME "ExampleModule2")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files of the library
set(SRCS
    src/Foo.cxx
    )

set(NO_DICT_SRCS # sources not for the dictionary
    src/Bar.cxx
    )

set(HEADERS # needed for the dictionary generation
    src/Bar.h
    include/${MODULE_NAME}/Foo.h
    )

set(LIBRARY_NAME ExampleModule2)
set(BUCKET_NAME ExampleModule2_Bucket)
set(LINKDEF src/ExampleLinkDef.h) # needed for the dictionary generation

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
