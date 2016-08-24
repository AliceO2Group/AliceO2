# Create a library called "libGen" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.
set(MODULE_NAME "Generators")

O2_SETUP(NAME ${MODULE_NAME})
if (PYTHIA8_INCLUDE_DIR)

  set(SRCS
      src/Pythia6Generator.cxx
      src/Pythia8Generator.cxx
      )

  set(HEADERS
      include/${MODULE_NAME}/Pythia6Generator.h
      include/${MODULE_NAME}/Pythia8Generator.h
      )

  set(LINKDEF src/GeneratorsLinkDef.h)
  set(LIBRARY_NAME ${MODULE_NAME})
    set(BUCKET_NAME generators_bucket)

#  set(DEPENDENCIES Base O2SimulationDataFormat pythia8 Pythia6)

  O2_GENERATE_LIBRARY()


else (PYTHIA8_INCLUDE_DIR)
  message(STATUS "module 'Generators' requires Pythia8 ... deactivated")
endif (PYTHIA8_INCLUDE_DIR)
