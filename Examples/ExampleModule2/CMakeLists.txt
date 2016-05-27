# @author  Barthélémy von Haller

set(MODULE_NAME "ExampleModule2")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/Foo.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Foo.h
    )

set(LIBRARY_NAME ExampleModule2)
set(BUCKET_NAME ExampleModule2_Bucket)
set(LINKDEF src/ExampleLinkDef.h)

O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)

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
