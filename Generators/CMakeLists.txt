# Create a library called "libGen" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.
set(MODULE_NAME "Generators")

FILE(GLOB INSTALL_SHARE "share/external/*.C")
INSTALL(FILES ${INSTALL_SHARE} DESTINATION share/Generators/external/)
SET(IGNORE_MACROS ${CMAKE_SOURCE_FILE}/share/Generators/external/hijing.C)

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/Generator.cxx
    src/GeneratorTGenerator.cxx
    src/GeneratorFromFile.cxx
    src/Pythia6Generator.cxx
    src/PDG.cxx	
    src/PrimaryGenerator.cxx
    src/InteractionDiamondParam.cxx
   )
set(HEADERS
    include/${MODULE_NAME}/Generator.h
    include/${MODULE_NAME}/GeneratorTGenerator.h
    include/${MODULE_NAME}/GeneratorFromFile.h
    include/${MODULE_NAME}/Pythia6Generator.h
    include/${MODULE_NAME}/PDG.h
    include/${MODULE_NAME}/PrimaryGenerator.h
    include/${MODULE_NAME}/InteractionDiamondParam.h
   )
if (HAVESIMULATION)
  set(HEADERS ${HEADERS}
      include/${MODULE_NAME}/GeneratorFactory.h
    )
  set(SRCS ${SRCS}
      src/GeneratorFactory.cxx
    )
endif()
 
if (PYTHIA8_INCLUDE_DIR)
  set(SRCS ${SRCS}
      src/Pythia8Generator.cxx
     )
  set(HEADERS ${HEADERS}
      include/${MODULE_NAME}/Pythia8Generator.h
      )
  set(BUCKET_NAME generators_bucket)
else (PYTHIA8_INCLUDE_DIR)
  message(STATUS "module 'Generators' requires Pythia8 ... deactivated")
  set(BUCKET_NAME generators_base_bucket)
endif (PYTHIA8_INCLUDE_DIR)

set(LINKDEF src/GeneratorsLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})

O2_GENERATE_LIBRARY()
