set(MODULE_NAME "DetectorsBase")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Detector.cxx
  src/GeometryManager.cxx
  src/MaterialManager.cxx
  src/Propagator.cxx
  src/MatLayerCyl.cxx
  src/MatLayerCylSet.cxx
  src/Ray.cxx
)

Set(HEADERS
  include/${MODULE_NAME}/Detector.h
  include/${MODULE_NAME}/GeometryManager.h
  include/${MODULE_NAME}/MaterialManager.h
  include/${MODULE_NAME}/Propagator.h
  include/${MODULE_NAME}/Triggers.h
  include/${MODULE_NAME}/Ray.h
  include/${MODULE_NAME}/MatCell.h
  include/${MODULE_NAME}/MatLayerCyl.h
  include/${MODULE_NAME}/MatLayerCylSet.h
)

Set(LINKDEF src/DetectorsBaseLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
set(BUCKET_NAME detectors_base_bucket)

O2_GENERATE_LIBRARY()

set(TEST_SRCS
  test/testMatBudLUT.cxx
)

if (HAVESIMULATION)
  O2_GENERATE_TESTS(
    MODULE_LIBRARY_NAME ${LIBRARY_NAME}
    BUCKET_NAME ${BUCKET_NAME}
    TEST_SRCS ${TEST_SRCS}
  )
endif()
