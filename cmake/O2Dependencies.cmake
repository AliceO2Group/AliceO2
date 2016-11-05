
########## DEPENDENCIES lookup ############

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

if (DDS_FOUND)
  add_definitions(-DENABLE_DDS)
  add_definitions(-DDDS_FOUND)
  set(OPTIONAL_DDS_LIBRARIES ${DDS_INTERCOM_LIBRARY_SHARED} ${DDS_PROTOCOL_LIBRARY_SHARED} ${DDS_USER_DEFAULTS_LIBRARY_SHARED})
  set(OPTIONAL_DDS_INCLUDE_DIR ${DDS_INCLUDE_DIR})
endif ()

# todo this should really not be needed. ROOT, Pythia, and FairRoot should comply with CMake best practices
# todo but they do not properly return DEPENDENCIES with absolute path.
link_directories(
    ${ROOT_LIBRARY_DIR}
    ${FAIRROOT_LIBRARY_DIR}
)
if(Pythia6_FOUND)
  link_directories(
      ${Pythia6_LIBRARY_DIR}
  )
endif()

########## General definitions and flags ##########

if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error") # avoid undefined in our libs
elseif(UNIX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined") # avoid undefined in our libs
endif()

########## Bucket definitions ############

o2_define_bucket(
    NAME
    common_boost_bucket

    SYSTEMINCLUDE_DIRECTORIES
    ${Boost_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    ExampleModule1_bucket

    DEPENDENCIES # library names
    common_boost_bucket
    ${Boost_PROGRAM_OPTIONS_LIBRARY}

    INCLUDE_DIRECTORIES
)

o2_define_bucket(
    NAME
    ExampleModule2_bucket

    DEPENDENCIES # library names
    ExampleModule1 # another module
    ExampleModule1_bucket # another bucket
    Core Hist # ROOT

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Examples/ExampleModule1/include # another module's include dir
)

o2_define_bucket(
    NAME
    flp2epn_bucket

    DEPENDENCIES
    common_boost_bucket
    ${Boost_CHRONO_LIBRARY}
    ${Boost_DATE_TIME_LIBRARY}
    ${Boost_LOG_LIBRARY}
    ${Boost_LOG_SETUP_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_RANDOM_LIBRARY}
    ${Boost_REGEX_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${ZMQ_LIBRARY_SHARED}
    ${OPTIONAL_DDS_LIBRARIES}
    Base
    FairTools
    FairMQ
    fairmq_logger
    pthread
    dl

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${OPTIONAL_DDS_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    flp2epndistrib_bucket

    DEPENDENCIES
    flp2epn_bucket
    ${OPTIONAL_DDS_LIBRARIES}

    INCLUDE_DIRECTORIES
    ${DDS_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    common_math_bucket

    DEPENDENCIES
    common_boost_bucket
    FairMQ ${Boost_LOG_LIBRARY} ${Boost_THREAD_LIBRARY} fairmq_logger Base FairTools Core MathCore Hist

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    common_field_bucket

    DEPENDENCIES
    Core RIO MathUtils Geom

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    CCDB_bucket

    DEPENDENCIES
    common_boost_bucket
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${Boost_LOG_SETUP_LIBRARY}
    ${Boost_LOG_LIBRARY}
    Base
    FairTools
    ParBase
    FairMQ ParMQ
    fairmq_logger pthread Core Tree XMLParser Hist Net RIO

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    root_base_bucket

    DEPENDENCIES
    Core # ROOT

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    fairroot_geom

    DEPENDENCIES
    Base GeoBase ParBase Geom Core
    common_boost_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    fairroot_base_bucket

    DEPENDENCIES
    root_base_bucket
    Base FairMQ FairTools ${Boost_LOG_LIBRARY} fairmq_logger Base
    common_boost_bucket
    ${Boost_THREAD_LIBRARY} pthread

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    root_physics_bucket

    DEPENDENCIES
    EG Physics  # ROOT

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
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

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    its_simulation_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_geom
    Hist
    Graf
    Gpad
    RIO
    fairroot_base_bucket
    root_physics_bucket
    ParBase
    ITSBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
)

o2_define_bucket(
    NAME
    hitanalysis_bucket

    DEPENDENCIES
    ITSSimulation

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    its_base_bucket

    DEPENDENCIES
    ParBase
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
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
    common_boost_bucket
    FairMQ
    fairmq_logger
    pthread
    ${Boost_LOG_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${Boost_LOG_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${OPTIONAL_DDS_LIBRARIES}

    INCLUDE_DIRECTORIES
    ${DDS_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${ZMQ_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    QC_apps_bucket

    DEPENDENCIES
    dl
    Core
    Base
    Hist
    FairMQ
    pthread
    fairmq_logger
    common_boost_bucket
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_LOG_LIBRARY}
    ${Boost_LOG_SETUP_LIBRARY}

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${ZMQ_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    QC_viewer_bucket

    DEPENDENCIES
    QC_apps_bucket
    Core
    RIO
    Net
    Gpad
)

o2_define_bucket(
    NAME
    QC_merger_bucket

    DEPENDENCIES
    QC_apps_bucket
)

o2_define_bucket(
    NAME
    QC_producer_bucket

    DEPENDENCIES
    QC_apps_bucket
    INCLUDE_DIRECTORIES
    ${ZMQ_INCLUDE_DIR}

   )

o2_define_bucket(
    NAME
    QC_test_bucket

    DEPENDENCIES
    dl Core Base Hist FairMQ  ${Boost_SYSTEM_LIBRARY}
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
    root_base_bucket
    fairroot_geom
    MathCore
    RIO
    TPCBase
    DetectorsBase
    SimulationDataFormat
    Generators

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
)

o2_define_bucket(
    NAME
    generators_bucket

    DEPENDENCIES
    Base SimulationDataFormat Pythia6 pythia8 MathCore

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${PYTHIA8_INCLUDE_DIR}
    ${PYTHIA6_INCLUDE_DIR}
)


o2_define_bucket(
    NAME
    alicehlt_bucket

    DEPENDENCIES
    dl
    ${CMAKE_THREAD_LIBS_INIT}
    ${FAIRMQ_DEPENDENCIES}
    common_boost_bucket
    ${Boost_CHRONO_LIBRARY}
    ${Boost_DATE_TIME_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_RANDOM_LIBRARY}
    ${Boost_REGEX_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_LOG_LIBRARY}
    ${Boost_LOG_SETUP_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    FairMQ
    ${OPTIONAL_DDS_LIBRARIES}

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${OPTIONAL_DDS_INCLUDE_DIR}

    SYSTEMINCLUDE_DIRECTORIES
    ${ZMQ_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    hough_bucket

    DEPENDENCIES
    Core RIO Gpad Hist HLTbase AliHLTUtil AliHLTTPC AliHLTUtil
    common_boost_bucket
    ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
    dl

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    mft_base_bucket

    DEPENDENCIES
    ParBase
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    mft_simulation_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_geom
    Hist
    Graf
    Gpad
    RIO
    XMLIO
    fairroot_base_bucket
    root_physics_bucket
    ParBase
    MFTBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
)

