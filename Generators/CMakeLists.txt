# Create a library called "libGen" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.
set(MODULE_NAME "Generators")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/GeneratorFromFile.cxx
    src/Pythia6Generator.cxx
    src/PDG.cxx	
   )
set(HEADERS
    include/${MODULE_NAME}/GeneratorFromFile.h
    include/${MODULE_NAME}/Pythia6Generator.h
    include/${MODULE_NAME}/PDG.h
   )

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
