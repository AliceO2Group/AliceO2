set(MODULE_NAME "DataFormatsMID")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
)

set(NO_DICT_HEADERS
  include/${MODULE_NAME}/StripPattern.h
)

set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME data_format_mid_bucket)

# The O2_GENERATE_LIBRARY does not work properly if there are no SRCS
# and there is no linkdef, since it fails in determining if is a C++ file.
# Let us install the headers directly
# Install all the public headers
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/${MODULE_NAME})
  install(DIRECTORY include/${MODULE_NAME} DESTINATION include)
endif()

# O2_GENERATE_LIBRARY()
