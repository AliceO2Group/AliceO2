# Create a library called "libGen" which includes the source files given in
# the array .
# The extension is already found.  Any number of sources could be listed here.
set(MODULE_NAME "Generators")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
    src/GeneratorFromFile.cxx
    src/PrimaryGenerator.cxx
    src/Generator.cxx
    src/GeneratorHepMC.cxx
    src/GeneratorTGenerator.cxx
    src/Trigger.cxx
    src/TriggerHepMC.cxx
    src/TriggerTGenerator.cxx
    src/Pythia6Generator.cxx
    src/Pythia8Generator.cxx
    src/PDG.cxx	
   )
set(HEADERS
    include/${MODULE_NAME}/GeneratorFromFile.h
    include/${MODULE_NAME}/PrimaryGenerator.h
    include/${MODULE_NAME}/Generator.h
    include/${MODULE_NAME}/GeneratorHepMC.h
    include/${MODULE_NAME}/GeneratorTGenerator.h
    include/${MODULE_NAME}/Trigger.h
    include/${MODULE_NAME}/TriggerHepMC.h
    include/${MODULE_NAME}/TriggerTGenerator.h
    include/${MODULE_NAME}/Pythia6Generator.h
    include/${MODULE_NAME}/Pythia8Generator.h
    include/${MODULE_NAME}/PDG.h
   )

if (NOT HepMC3_FOUND)
   list(REMOVE_ITEM SRCS
                    src/GeneratorHepMC.cxx
                    src/TriggerHepMC.cxx
                    )
   list(REMOVE_ITEM HEADERS
                    include/${MODULE_NAME}/GeneratorHepMC.h
                    include/${MODULE_NAME}/TriggerHepMC.h
                    )
   message(STATUS "module 'Generators' requires HepMC3 ... HepMC3 support deactivated")
endif()

if (NOT PYTHIA8_FOUND)
   list(REMOVE_ITEM SRCS
                    src/Pythia8Generator.cxx
                    )
   list(REMOVE_ITEM HEADERS
                    include/${MODULE_NAME}/Pythia8Generator.h
                    )
   message(STATUS "module 'Generators' requires Pythia8 ... Pythia8 support deactivated")
endif()

if (HAVESIMULATION)
  set(HEADERS ${HEADERS}
      include/${MODULE_NAME}/GeneratorFactory.h
    )
  set(SRCS ${SRCS}
      src/GeneratorFactory.cxx
    )
endif()
 
set(BUCKET_NAME generators_bucket)
set(LINKDEF src/GeneratorsLinkDef.h)
set(LIBRARY_NAME ${MODULE_NAME})

O2_GENERATE_LIBRARY()
