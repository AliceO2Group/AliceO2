# @author  Barthélémy von Haller

O2_SETUP(NAME "ExampleModule1")

# Define the source and header files
set(SRCS
    src/Foo.cxx
    )

set(HEADERS
    include/ExampleModule1/Foo.h
    )

set(LIBRARY_NAME ExampleModule1)
set(BUCKET_NAME ExampleModule1_Bucket)

O2_GENERATE_LIBRARY()

# todo we repeat ourselves because the above macro dares deleting the variables we pass to it.
set(LIBRARY_NAME ExampleModule1)
set(BUCKET_NAME ExampleModule1_Bucket)

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runExampleModule1
    SOURCES src/main.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)
