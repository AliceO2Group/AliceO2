
########## Dependencies lookup ############

find_package(ROOT 6.06.00 REQUIRED)
find_package(Pythia8)
find_package(Pythia6)
if (ALICEO2_MODULAR_BUILD)
  # Geant3, Geant4 installed via cmake
  find_package(Geant3)
  find_package(Geant4)
else (ALICEO2_MODULAR_BUILD)
  # For old versions of VMC packages (to be removed)
  find_package(GEANT3)
  find_package(GEANT4)
  find_package(GEANT4DATA)
  find_package(GEANT4VMC)
  find_package(CLHEP)
endif (ALICEO2_MODULAR_BUILD)
find_package(CERNLIB)
find_package(HEPMC)
find_package(IWYU)
find_package(DDS)

find_package(Boost 1.59 COMPONENTS thread system timer program_options random filesystem chrono exception regex serialization log log_setup unit_test_framework REQUIRED)

find_package(AliRoot)
find_package(FairRoot)

include_directories(SYSTEM
    ${BASE_INCLUDE_DIRECTORIES}
    ${Boost_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${ZMQ_INCLUDE_DIR}
    )

# todo this should really not be needed. ROOT and FairRoot should comply with CMake best practices but they do not properly return libraries with full path.
set(LINK_DIRECTORIES
    ${ROOT_LIBRARY_DIR}
    ${FAIRROOT_LIBRARY_DIR}
    )
link_directories(${LINK_DIRECTORIES})

########## Bucket definitions ############

o2_define_bucket(
    NAME
    ExampleModule1_Bucket
    DEPENDENCIES # library names
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

o2_define_bucket(
    NAME
    ExampleModule2_Bucket
    DEPENDENCIES # library names
    ExampleModule1_Bucket
    Core Hist # ROOT
)

o2_define_bucket(
    NAME
    CCDB_Bucket
    DEPENDENCIES
    Base ParBase FairMQ ParMQ ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_LOG_LIBRARY} fairmq_logger pthread
    Core Tree XMLParser Hist # ROOT
)

o2_define_bucket(
    NAME
    flp2epn_Bucket
    DEPENDENCIES
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_DATE_TIME_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_CHRONO_LIBRARY} FairMQ ${Boost_LOG_LIBRARY} fairmq_logger
)