set(MODULE_NAME "FITBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Geometry.cxx
  src/Digit.cxx
    )

set(HEADERS
    include/${MODULE_NAME}/Geometry.h
    include/${MODULE_NAME}/Digit.h
    )


Set(LINKDEF src/FITBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME fit_base_bucket)

O2_GENERATE_LIBRARY()

install(
  DIRECTORY files
  DESTINATION share/Detectors/FIT/
)
