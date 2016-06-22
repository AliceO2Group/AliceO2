
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

find_package(Boost 1.59 COMPONENTS thread system timer program_options random filesystem chrono exception regex serialization log log_setup unit_test_framework date_time REQUIRED)
find_package(ZeroMQ)

find_package(AliRoot)
find_package(FairRoot REQUIRED)
find_package(FairMQ REQUIRED)

include_directories(SYSTEM
    ${BASE_INCLUDE_DIRECTORIES}
    ${Boost_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${ZMQ_INCLUDE_DIR}
    )

if (DDS_FOUND)
  add_definitions(-DENABLE_DDS)

  include_directories(SYSTEM
      ${DDS_INCLUDE_DIR}
      )
endif ()

# todo this should really not be needed. ROOT and FairRoot should comply with CMake best practices but they do not properly return libraries with full path.
set(LINK_DIRECTORIES
    ${ROOT_LIBRARY_DIR}
    ${FAIRROOT_LIBRARY_DIR}
    )
link_directories(${LINK_DIRECTORIES})

########## General definitions and flags ##########

if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error") # avoid undefined in our libs
elseif(UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined") # avoid undefined in our libs
endif() 

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
    ExampleModule1 # another module
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    Core Hist # ROOT
)

o2_define_bucket(
    NAME
    CCDB_Bucket
    DEPENDENCIES
    Base FairTools FairMQ ParBase ParMQ ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_LOG_LIBRARY} ${Boost_THREAD_LIBRARY} fairmq_logger pthread
    Core Tree XMLParser Hist Net RIO # ROOT
)

o2_define_bucket(
    NAME
    flp2epn_Bucket
    DEPENDENCIES
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_DATE_TIME_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_CHRONO_LIBRARY} ${Boost_DATE_TIME_LIBRARY}
    ${ZMQ_LIBRARIES} Base FairTools FairMQ ${Boost_LOG_LIBRARY} fairmq_logger pthread
)

o2_define_bucket(
    NAME
    flp2epndistrib_Bucket
    DEPENDENCIES
    flp2epndistrib_no_dds_Bucket
    dds-key-value-lib
)

o2_define_bucket(
    NAME
    common_math_bucket
    DEPENDENCIES
    FairMQ ${Boost_LOG_LIBRARY} ${Boost_THREAD_LIBRARY} fairmq_logger Base FairTools Core MathCore Hist
)

o2_define_bucket(
    NAME
    common_field_bucket
    DEPENDENCIES
    Core RIO MathUtils Geom
)

o2_define_bucket(
    NAME
    root_base_bucket
    DEPENDENCIES
    Core # ROOT
)

o2_define_bucket(
    NAME
    root_geom
    DEPENDENCIES
    Base GeoBase ParBase Geom Core
)

o2_define_bucket(
    NAME
    fairroot_base_bucket
    DEPENDENCIES
    root_base_bucket
    Base FairMQ FairTools ${Boost_LOG_LIBRARY} fairmq_logger Base
    ${Boost_THREAD_LIBRARY} pthread
)

o2_define_bucket(
    NAME
    root_physics_bucket
    DEPENDENCIES
    EG Physics  # ROOT
)

o2_define_bucket(
    NAME
    data_format_simulation_bucket
    DEPENDENCIES
    fairroot_base_bucket
    root_physics_bucket
)

o2_define_bucket(
    NAME
    detectors_base
    DEPENDENCIES
    fairroot_base_bucket
    root_physics_bucket
    VMC # ROOT
)

o2_define_bucket(
    NAME
    its_simulation_bucket
    DEPENDENCIES
    root_base_bucket
    root_geom
    Hist
    Graf
    Gpad
    RIO
    fairroot_base_bucket
    root_physics_bucket
    ParBase
    itsBase
    DetectorsBase
    SimulationDataFormat
)

o2_define_bucket(
    NAME
    itsmft_test
    DEPENDENCIES
    itsSimulation
)

o2_define_bucket(
    NAME
    its_base_bucket
    DEPENDENCIES
    ParBase
    DetectorsBase
)

o2_define_bucket(
    NAME
    QC_base_bucket
    DEPENDENCIES
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    RIO
    Core
    MathMore
    Net
    Hist
    Tree
    Gpad
    MathCore
    ${Boost_LOG_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    FairMQ ${Boost_THREAD_LIBRARY} ${Boost_LOG_LIBRARY} fairmq_logger
    pthread
)

o2_define_bucket(
    NAME
    QC_viewer_bucket
    DEPENDENCIES
    QC_base_bucket
    Gpad
)

o2_define_bucket(
    NAME
    QC_merger_bucket
    DEPENDENCIES
    QC_base_bucket
)

o2_define_bucket(
    NAME
    QC_producer_bucket
    DEPENDENCIES
    QC_base_bucket
)

o2_define_bucket(
    NAME
    QC_test_bucket
    DEPENDENCIES
    dl Core Base Hist o2qaLibrary FairMQ  ${Boost_SYSTEM_LIBRARY}
)

o2_define_bucket(
    NAME
    tpc_base_bucket
    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    ParBase
)

o2_define_bucket(
    NAME
    tpc_simulation_bucket
    DEPENDENCIES
    TPCbase
    DetectorsBase
    SimulationDataFormat
)

o2_define_bucket(
    NAME
    generators_bucket
    DEPENDENCIES
    Base O2SimulationDataFormat pythia8 Pythia6
)

o2_define_bucket(
    NAME
    alicehlt_bucket
    DEPENDENCIES
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_CHRONO_LIBRARY}
    fairroot_base_bucket
)

o2_define_bucket(
    NAME
    alicehlt_with_dds_bucket
    DEPENDENCIES
    alicehlt_bucket
    dds-key-value-lib
)

o2_define_bucket(
    NAME
    hough_Bucket
    DEPENDENCIES
    Core RIO Gpad Hist HLTbase AliHLTTPC AliHLTUtil ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} dl
)