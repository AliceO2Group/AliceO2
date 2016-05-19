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

# extra variables to generate a dictionary (should go away if O2_GENERATE_LIBRARY takes arguments instead)
set(INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
set(LINKDEF src/ExampleLinkDef.h)

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
