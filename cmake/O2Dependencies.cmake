
########## DEPENDENCIES lookup ############

find_package(ROOT 6.06.00 REQUIRED)
find_package(Vc REQUIRED)
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

find_package(Boost 1.59 COMPONENTS thread system timer program_options random filesystem chrono exception regex serialization log log_setup unit_test_framework date_time signals REQUIRED)
# for the guideline support library
include_directories(${MS_GSL_INCLUDE_DIR})

find_package(AliRoot)
find_package(FairRoot REQUIRED)
find_package(FairMQ REQUIRED)
find_package(Protobuf REQUIRED)

find_package(GLFW)

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
    ${Boost_LIBRARY_DIRS}
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
  glfw_bucket

  DEPENDENCIES
  ${GLFW_LIBRARIES}

  INCLUDE_DIRECTORIES
  ${GLFW_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    common_vc_bucket

    DEPENDENCIES
    ${Vc_LIBRARIES}

    INCLUDE_DIRECTORIES
    ${Vc_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    common_boost_bucket

    DEPENDENCIES
    Boost::system
    Boost::log
    Boost::log_setup
    Boost::program_options
    Boost::thread

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
    Boost::chrono
    Boost::date_time
    Boost::random
    Boost::regex
    Base
    Headers
    FairTools
    FairRoot::FairMQ
    pthread
    dl

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
)

# a common bucket for the implementation of devices inherited
# from O2device
if(GLFW_FOUND)
    set(GUI_LIBRARIES DebugGUI)
endif()

o2_define_bucket(
    NAME
    O2DeviceApplication_bucket

    DEPENDENCIES
    Base
    Headers
    TimeFrame
    O2Device
    dl
)

o2_define_bucket(
    NAME
    O2FrameworkCore_bucket
    DEPENDENCIES
    O2DeviceApplication_bucket
    Core
    Net
    ${GUI_LIBRARIES}
)

o2_define_bucket(
    NAME
    FrameworkApplication_bucket

    DEPENDENCIES
    O2FrameworkCore_bucket
    Framework
    Hist
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
    DataFormatsTPC_bucket
    
    DEPENDENCIES
    tpc_base_bucket
    
    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
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
    ${FAIRROOT_INCLUDE_DIR}/fairmq # temporary fix, until bucket system works with imported targets
    ${CMAKE_SOURCE_DIR}/DataFormats/Headers/include
)

o2_define_bucket(
    NAME
    O2DataProcessingApplication_bucket

    DEPENDENCIES
    O2DeviceApplication_bucket
    Framework
    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Framework/Core/include
)

o2_define_bucket(
    NAME
    flp2epn_bucket

    DEPENDENCIES
    common_boost_bucket
    Boost::chrono
    Boost::date_time
    Boost::random
    Boost::regex
    Base
    Headers
    FairTools
    FairRoot::FairMQ
    pthread
    dl

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    flp2epndistrib_bucket

    DEPENDENCIES
    flp2epn_bucket

    INCLUDE_DIRECTORIES
)

o2_define_bucket(
    NAME
    common_math_bucket

    DEPENDENCIES
    common_boost_bucket
    FairRoot::FairMQ Base FairTools Core MathCore Matrix Minuit Hist Geom GenVector RIO

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
    configuration_bucket

    DEPENDENCIES
    common_boost_bucket
    root_base_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    CCDB_bucket

    DEPENDENCIES
    dl
    common_boost_bucket
    Boost::filesystem
    ${PROTOBUF_LIBRARY}
    Base
    FairTools
    ParBase
    ParMQ
    FairRoot::FairMQ pthread Core Tree XMLParser Hist Net RIO z

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}/fairmq
    ${ROOT_INCLUDE_DIR}

    SYSTEMINCLUDE_DIRECTORIES
    ${PROTOBUF_INCLUDE_DIR}
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
    FairTools
    Base GeoBase ParBase Geom Core VMC Tree
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
    fairroot_geom
    Base
    FairTools
    FairRoot::FairMQ
    common_boost_bucket
    Boost::thread
    pthread

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
    common_math_bucket
    detectors_base_bucket
    DetectorsBase
    RIO

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    detectors_base_bucket

    DEPENDENCIES
    fairroot_base_bucket
    root_physics_bucket
    Field
    VMC # ROOT
    Geom

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Common/Field/include
    ${CMAKE_SOURCE_DIR}/Common/Constants/include
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
    SimulationDataFormat
    detectors_base_bucket
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/Base/include
)

o2_define_bucket(
    NAME
    mcsteplogger_bucket

    DEPENDENCIES
    dl
    root_base_bucket
    VMC
    EG
    Tree
    Geom

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
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
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
    DetectorsBase
    SimulationDataFormat
    
    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${MS_GSL_INCLUDE_DIR}
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
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include

    SYSTEMINCLUDE_DIRECTORIES
    ${Boost_INCLUDE_DIR}
    )

o2_define_bucket(
    NAME
    QC_base_bucket

    DEPENDENCIES
    ${CMAKE_THREAD_LIBS_INIT}
    Boost::unit_test_framework
    RIO
    Core
    MathMore
    Net
    Hist
    Tree
    Gpad
    MathCore
    common_boost_bucket
    FairRoot::FairMQ
    pthread
    Boost::date_time
    Boost::timer
    ${OPTIONAL_DDS_LIBRARIES}

    INCLUDE_DIRECTORIES
    ${DDS_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    QC_apps_bucket

    DEPENDENCIES
    dl
    Core
    Base
    Hist
    pthread
    FairRoot::FairMQ
    common_boost_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
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
   )

o2_define_bucket(
    NAME
    QC_workflow_bucket

    DEPENDENCIES
    QCProducer
    QCMerger
    Framework

    INCLUDE_DIRECTORIES
   )

o2_define_bucket(
    NAME
    QC_test_bucket

    DEPENDENCIES
    dl Core Base Hist FairRoot::FairMQ
    common_boost_bucket
)

o2_define_bucket(
    NAME
    tpc_base_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    common_vc_bucket
    common_math_bucket
    ParBase
    MathUtils
    Core Hist Gpad

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    tpc_simulation_bucket

    DEPENDENCIES
    tpc_base_bucket
    Field
    DetectorsBase
    Generators
    TPCBase
    SimulationDataFormat
    Geom
    MathCore
    MathUtils
    RIO
    Hist
    DetectorsPassive
    Gen
    Base
    TreePlayer
    Steer
    #   Core
    #    root_base_bucket
    #    fairroot_geom
    #    ${GENERATORS_LIBRARY}

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/Passive/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/Field/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${MS_GSL_INCLUDE_DIR}
)


o2_define_bucket(
    NAME
    tpc_reconstruction_bucket

    DEPENDENCIES
    tpc_base_bucket
    DataFormatsTPC_bucket
    DetectorsBase
    TPCBase
    SimulationDataFormat
    Geom
    MathCore
    RIO
    Hist
    DetectorsPassive
    Gen
    Base
    TreePlayer
    TPCSimulation
    #the dependency on TPCSimulation should be removed at some point
    #perhaps 'Cluster' can be moved to base, or so

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/TPC/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/Passive/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    tpc_calibration_bucket

    DEPENDENCIES
    tpc_base_bucket
    tpc_reconstruction_bucket
    DetectorsBase
    TPCBase
    TPCReconstruction
    MathUtils

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    tpc_monitor_bucket

    DEPENDENCIES
    tpc_calibration_bucket
    tpc_base_bucket
    tpc_reconstruction_bucket
    DetectorsBase
    TPCBase
    TPCCalibration
    TPCReconstruction

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/calibration/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/reconstruction/include
)

# base bucket for generators not needing any external stuff
o2_define_bucket(
    NAME
    generators_base_bucket

    DEPENDENCIES
    Base SimulationDataFormat MathCore RIO Tree
    fairroot_base_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    generators_bucket

    DEPENDENCIES
    generators_base_bucket
    pythia8

    INCLUDE_DIRECTORIES
    ${PYTHIA8_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    hough_bucket

    DEPENDENCIES
    Core RIO Gpad Hist HLTbase AliHLTUtil AliHLTTPC AliHLTUtil
    common_boost_bucket
    Boost::filesystem
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
    ITSMFTSimulation
    DetectorsBase
    Graf
    Gpad
    XMLIO

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include

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
    DetectorsBase
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

o2_define_bucket(
    NAME
    emcal_base_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    Geom
    MathCore
    Matrix
    Physics
    ParBase
    VMC
    Geom
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    passive_detector_bucket
 
    DEPENDENCIES
    fairroot_geom
    Field
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/Field/include
    ${CMAKE_SOURCE_DIR}/Detectors/Passive/include
)

o2_define_bucket(
    NAME
    emcal_simulation_bucket

    DEPENDENCIES
    emcal_base_bucket
    root_base_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    EMCALBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/EMCAL/base/include
)

o2_define_bucket(
    NAME
    tof_simulation_bucket

    DEPENDENCIES
    emcal_base_bucket
    root_base_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    TOFBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TOF/base/include
)
o2_define_bucket(
    NAME
    fit_base_bucket

    DEPENDENCIES # library names
    root_base_bucket
    fairroot_geom
    root_base_bucket
    fairroot_base_bucket
    Matrix
    Physics
    Geom
    Core Hist # ROOT

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/base/include
 )
o2_define_bucket(
    NAME
    fit_simulation_bucket

    DEPENDENCIES # library names
    root_base_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    FITBase
    DetectorsBase
    SimulationDataFormat
    Core Hist # ROOT

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/Simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/Simulations/include 
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    phos_base_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    Geom
    MathCore
    Matrix
    Physics
    ParBase
    VMC
    Geom
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    phos_simulation_bucket

    DEPENDENCIES
    phos_base_bucket
    root_base_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    PHOSBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/PHOS/base/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    event_visualisation_base_bucket

    DEPENDENCIES
    root_base_bucket
    EventVisualisationDataConverter
    Graf3d
    Eve
    RGL
    Gui
    CCDB

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/CCDB/include
    ${CMAKE_SOURCE_DIR}/EventVisualisation/DataConverter/include

    SYSTEMINCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    trd_simulation_bucket

    DEPENDENCIES
    emcal_base_bucket
    root_base_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    TRDBase
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/include
)

# a bucket for "global" executables/macros
o2_define_bucket(
    NAME
    run_bucket

    DEPENDENCIES
    #-- buckets follow
    fairroot_base_bucket
    pythia8

    #-- precise modules follow
    SimConfig
    DetectorsPassive
    TPCSimulation
    TPCReconstruction
    ITSSimulation
    MFTSimulation
    TRDSimulation
    EMCALSimulation
    TOFSimulation
    FITSimulation
    PHOSSimulation
    Field
    Generators
    DataFormatsParameters
)

o2_define_bucket(
NAME
    event_visualisation_detectors_bucket

    DEPENDENCIES
    root_base_bucket
    EventVisualisationBase
    EventVisualisationDataConverter
    Graf3d
    Eve
    RGL
    Gui
    CCDB

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/EventVisualisation/Base/include
    ${CMAKE_SOURCE_DIR}/EventVisualisation/DataConverter/include

    SYSTEMINCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    event_visualisation_view_bucket

    DEPENDENCIES
    root_base_bucket
    EventVisualisationBase
    EventVisualisationDetectors
    EventVisualisationDataConverter
    Graf3d
    Eve
    RGL
    Gui
    CCDB

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/EventVisualisation/Base/include
    ${CMAKE_SOURCE_DIR}/EventVisualisation/Detectors/include
    ${CMAKE_SOURCE_DIR}/EventVisualisation/DataConverter/include

    SYSTEMINCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
NAME
    event_visualisation_data_converter_bucket

    DEPENDENCIES
    root_base_bucket
    Graf3d
    Eve
    RGL
    Gui
    CCDB

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/CCDB/include

    SYSTEMINCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    Algorithm_bucket

    DEPENDENCIES
    Headers

    INCLUDE_DIRECTORIES
)

o2_define_bucket(
  NAME
  data_parameters_bucket

  DEPENDENCIES
  Core
  detectors_base_bucket
  DetectorsBase
  
  INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/Detectors/Base/include
  ${CMAKE_SOURCE_DIR}/Common/Constants/include
  ${CMAKE_SOURCE_DIR}/Common/Types/include
)

o2_define_bucket(
  NAME
  common_utils_bucket

  DEPENDENCIES
  Core Tree
  DetectorsBase # for test dependency only
  
  INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/Detectors/Base/include
)
