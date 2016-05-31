set(MODULE_NAME "itsSimulation")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/UpgradeGeometryTGeo.cxx
    src/V11Geometry.cxx
    src/UpgradeV1Layer.cxx
    src/Segmentation.cxx
    src/UpgradeSegmentationPixel.cxx
    src/GeometryManager.cxx
    src/Detector.cxx
    src/GeometryHandler.cxx
    src/Point.cxx
    src/Chip.cxx
    src/Digit.cxx
    src/Digitizer.cxx
    src/DigitizerTask.cxx
    src/DigitWriteoutBuffer.cxx
    src/DigitContainer.cxx
    src/DigitLayer.cxx
    src/DigitStave.cxx
    )
set(HEADERS
    include/${MODULE_NAME}/Chip.h
    include/${MODULE_NAME}/Detector.h
    include/${MODULE_NAME}/Digit.h
    include/${MODULE_NAME}/DigitContainer.h
    include/${MODULE_NAME}/DigitLayer.h
    include/${MODULE_NAME}/DigitStave.h
    include/${MODULE_NAME}/DigitWriteoutBuffer.h
    include/${MODULE_NAME}/Digitizer.h
    include/${MODULE_NAME}/DigitizerTask.h
    include/${MODULE_NAME}/GeometryHandler.h
    include/${MODULE_NAME}/GeometryManager.h
    include/${MODULE_NAME}/Point.h
    include/${MODULE_NAME}/Segmentation.h
    include/${MODULE_NAME}/UpgradeGeometryTGeo.h
    include/${MODULE_NAME}/UpgradeSegmentationPixel.h
    include/${MODULE_NAME}/UpgradeV1Layer.h
    include/${MODULE_NAME}/V11Geometry.h
    )

Set(LINKDEF src/itsSimulationLinkDef.h)
Set(LIBRARY_NAME itsSimulation)
set(BUCKET_NAME its_simulation_bucket)

O2_GENERATE_LIBRARY()

# Install all the public header
install(DIRECTORY include/${MODULE_NAME} DESTINATION include)