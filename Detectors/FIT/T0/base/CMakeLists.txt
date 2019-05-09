set(MODULE_NAME "T0Base")
set(BUCKET_NAME fit_base_bucket)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Geometry.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Geometry.h
    )

  Set(LINKDEF src/T0BaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})


O2_GENERATE_LIBRARY()

install(
  DIRECTORY files
  DESTINATION share/Detectors/T0/
)
