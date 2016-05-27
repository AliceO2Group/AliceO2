# @author  Barthélémy von Haller

set(MODULE_NAME "ExampleModule1")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/Foo.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Foo.h
    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ExampleModule1_Bucket)

O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ExampleModule1_Bucket)

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runExampleModule1
    SOURCES src/main.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)
