set(MODULE_NAME "V0Base")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Geometry.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Geometry.h
    )

Set(LINKDEF src/V0BaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fit_base_bucket)

O2_GENERATE_LIBRARY()

install(
  DIRECTORY files
  DESTINATION share/Detectors/V0/
)
