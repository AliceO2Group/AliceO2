# @author  Barthélémy von Haller

set(MODULE_NAME "ExampleModule1")

O2_SETUP(NAME ${MODULE_NAME})

# Define the source and header files
set(SRCS
    src/Foo.cxx
    src/Bar.cxx
    )

# HEADERS is not needed if we don't generate a dictionary.
#set(HEADERS
#    include/${MODULE_NAME}/Foo.h
#    src/Bar.h
#    )

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME ExampleModule1_bucket)

O2_GENERATE_LIBRARY()

# Define application
O2_GENERATE_EXECUTABLE(
    EXE_NAME runExampleModule1
    SOURCES src/main.cxx
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
)



set(TEST_SRCS
    test/testExampleModule1.cxx
    )

O2_GENERATE_TESTS(
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
    TEST_SRCS ${TEST_SRCS}
)
