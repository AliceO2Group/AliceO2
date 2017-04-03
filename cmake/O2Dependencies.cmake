
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
set (ZeroMQ_NO_DEPRECATED 1) # no deprecation warning since we have converted to new variables
find_package(ZeroMQ)


find_package(AliRoot)
find_package(FairRoot REQUIRED)
find_package(FairMQ REQUIRED)
find_package(Protobuf REQUIRED)

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
if(PYTHIA8_FOUND)
  link_directories(
      ${PYTHIA8_LIB_DIR}
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

    DEPENDENCIES
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_LOG_LIBRARY}
    ${Boost_LOG_SETUP_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_THREAD_LIBRARY}

    SYSTEMINCLUDE_DIRECTORIES
    ${Boost_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    ExampleModule1_bucket

    DEPENDENCIES # library names and other buckets
    common_boost_bucket

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
    O2Device_bucket

    DEPENDENCIES
    common_boost_bucket
    ${Boost_CHRONO_LIBRARY}
    ${Boost_DATE_TIME_LIBRARY}
    ${Boost_RANDOM_LIBRARY}
    ${Boost_REGEX_LIBRARY}
    ${ZeroMQ_LIBRARY_SHARED}
    ${OPTIONAL_DDS_LIBRARIES}
    Base
    Headers
    FairTools
    FairMQ
    fairmq_logger
    pthread
    dl

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${OPTIONAL_DDS_INCLUDE_DIR}
)

# a common bucket for the implementation of devices inherited
# from O2device
o2_define_bucket(
    NAME
    O2DeviceApplication_bucket

    DEPENDENCIES
    Base
    Headers
    O2Device
    dl
)

o2_define_bucket(
    NAME
    O2MessageMonitor_bucket

    DEPENDENCIES
    O2Device_bucket
    O2Device
)

o2_define_bucket(
    NAME
    TimeFrame_bucket

    DEPENDENCIES
    Base
    Headers
    fairroot_base_bucket

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ZeroMQ_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/Headers/include
)

o2_define_bucket(
    NAME
    flp2epn_bucket

    DEPENDENCIES
    common_boost_bucket
    ${Boost_CHRONO_LIBRARY}
    ${Boost_DATE_TIME_LIBRARY}
    ${Boost_RANDOM_LIBRARY}
    ${Boost_REGEX_LIBRARY}
    ${ZeroMQ_LIBRARY_SHARED}
    ${OPTIONAL_DDS_LIBRARIES}
    Base
    Headers
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
    FairMQ fairmq_logger Base FairTools Core MathCore Hist

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    common_field_bucket

    DEPENDENCIES
    fairroot_base_bucket
    Base ParBase Core RIO MathUtils Geom

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    CCDB_bucket

    DEPENDENCIES
    dl
    common_boost_bucket
    ${Boost_FILESYSTEM_LIBRARY}
    ${PROTOBUF_LIBRARY}
    Base
    FairTools
    ParBase
    FairMQ ParMQ
    fairmq_logger pthread Core Tree XMLParser Hist Net RIO z

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ZeroMQ_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    root_base_bucket

    DEPENDENCIES
    Core RIO GenVector # ROOT

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
    ${ZeroMQ_LIBRARY_SHARED}
    Base FairMQ FairTools fairmq_logger Base
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
    ${ROOT_INCLUDE_DIR}/Math/GenVector
)

o2_define_bucket(
    NAME
    itsmft_base_bucket

    DEPENDENCIES
    fairroot_base_bucket
    MathCore
    Geom
    RIO
    Hist
    ParBase
    Field

    INCLUDE_DIRECTORIES
)

o2_define_bucket(
    NAME
    itsmft_simulation_bucket

    DEPENDENCIES
    itsmft_base_bucket
    Graf
    Gpad
    DetectorsBase
    SimulationDataFormat
    ITSMFTBase

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
)

o2_define_bucket(
    NAME
    itsmft_reconstruction_bucket

    DEPENDENCIES
    itsmft_base_bucket
    Graf
    Gpad
    DetectorsBase
    ITSMFTBase

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
)

o2_define_bucket(
    NAME
    its_base_bucket

    DEPENDENCIES
    itsmft_base_bucket
    ITSMFTBase

    INCLUDE_DIRECTORIES
)

o2_define_bucket(
    NAME
    its_simulation_bucket

    DEPENDENCIES
    its_base_bucket
    Graf
    Gpad
    ITSMFTBase
    ITSMFTSimulation
    ITSBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/include
)

o2_define_bucket(
    NAME
    its_reconstruction_bucket

    DEPENDENCIES
    its_base_bucket
    ITSMFTBase
    ITSMFTReconstruction
    ITSBase
    ITSSimulation
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/simulation/include
)

o2_define_bucket(
    NAME
    hitanalysis_bucket

    DEPENDENCIES
    ITSSimulation

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/include
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
    ${Boost_DATE_TIME_LIBRARY}
    ${OPTIONAL_DDS_LIBRARIES}

    INCLUDE_DIRECTORIES
    ${DDS_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${ZeroMQ_INCLUDE_DIR}
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

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${ZeroMQ_INCLUDE_DIR}
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
    ${ZeroMQ_INCLUDE_DIR}

   )

o2_define_bucket(
    NAME
    QC_test_bucket

    DEPENDENCIES
    dl Core Base Hist FairMQ
    common_boost_bucket
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
    ${GENERATORS_LIBRARY}

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
)

o2_define_bucket(
    NAME
    generators_bucket

    DEPENDENCIES
    Base SimulationDataFormat pythia6 pythia8 MathCore

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
    ${Boost_RANDOM_LIBRARY}
    ${Boost_REGEX_LIBRARY}
    FairMQ
    ${OPTIONAL_DDS_LIBRARIES}

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${OPTIONAL_DDS_INCLUDE_DIR}

    SYSTEMINCLUDE_DIRECTORIES
    ${ZeroMQ_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    hough_bucket

    DEPENDENCIES
    Core RIO Gpad Hist HLTbase AliHLTUtil AliHLTTPC AliHLTUtil
    common_boost_bucket
    ${Boost_FILESYSTEM_LIBRARY}
    dl

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    mft_base_bucket

    DEPENDENCIES
    itsmft_base_bucket
    ITSMFTBase
    Graf
    Gpad
    XMLIO

    INCLUDE_DIRECTORIES

)

o2_define_bucket(
    NAME
    mft_simulation_bucket

    DEPENDENCIES
    mft_base_bucket
    ITSMFTBase
    ITSMFTSimulation
    MFTBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/base/include

)

o2_define_bucket(
    NAME
    mft_reconstruction_bucket

    DEPENDENCIES
    mft_base_bucket
    ITSMFTBase
    ITSMFTReconstruction
    MFTBase
    MFTSimulation
    O2Device_bucket
    Tree
    Net

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/simulation/include

)

