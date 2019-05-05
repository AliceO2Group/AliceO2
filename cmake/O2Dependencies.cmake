
########## DEPENDENCIES lookup ############

function(guess_append_libpath _libname _root)
  # Globally adds, as library path, the path of library ${_libname} searched
  # under ${_root}/lib and ${_root}/lib64. The purpose is to work around broken
  # external CMake config files, hardcoding full paths of their dependencies
  # not being relocated properly, leading to broken builds if reusing builds
  # produced under different hosts/paths.
  unset(_lib CACHE)  # force find_library to look again
  find_library(_lib "${_libname}" HINTS "${_root}" "${_root}/.." NO_DEFAULT_PATH PATH_SUFFIXES lib lib64)
  if(_lib)
    get_filename_component(_libdir "${_lib}" DIRECTORY)
    message(STATUS "Adding library path: ${_libdir}")
    link_directories(${_libdir})
  else()
    message(WARNING "Cannot find library ${_libname} under ${_root}")
  endif()
endfunction()

find_package(ROOT 6.06.00 REQUIRED)
find_package(Vc REQUIRED)
find_package(Pythia8)
find_package(Pythia6)

# Installed via CMake. Note: we work around hardcoded full paths in the CMake
# config files not being relocated properly by appending library paths.
guess_append_libpath(geant321 "${Geant3_DIR}")
find_package(Geant3 NO_MODULE)
guess_append_libpath(G4run "${Geant4_DIR}")
find_package(Geant4 NO_MODULE)
guess_append_libpath(geant4vmc "${GEANT4_VMC_DIR}")
find_package(Geant4VMC NO_MODULE)
guess_append_libpath(BaseVGM "${VGM_DIR}")

find_package(VGM NO_MODULE)
find_package(CERNLIB)
find_package(HEPMC)
# FIXME: the way, iwyu is integrated now conflicts with the possibility to add
# custom rules for individual modules, e.g. the custom targets introduced in
# PR #886 depending on some header files conflict with the IWYU setup
# disable package for the moment
#find_package(IWYU)

find_package(Boost 1.59 COMPONENTS container thread system timer program_options random filesystem chrono exception regex serialization log log_setup unit_test_framework date_time signals iostreams REQUIRED)
# for the guideline support library
include_directories(${MS_GSL_INCLUDE_DIR})

find_package(AliRoot)
find_package(FairRoot REQUIRED)
find_package(FairMQInFairRoot) # DEPRECATED: This looks for FairMQ embedded in old FairRoot versions,
                               # before FairMQ and FairLogger have moved to separate repos.
                               # Remove this line, once we require FairMQ 1.2+.
if(NOT FairMQInFairRoot_FOUND) # DEPRECATED: Remove this condition, once we require FairMQ 1.2+
  find_package(FairMQ REQUIRED)
  find_package(FairLogger REQUIRED)
endif()
find_package(DDS)
cmake_policy(SET CMP0077 NEW)
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(protobuf CONFIG REQUIRED)
find_package(InfoLogger REQUIRED)
find_package(Configuration REQUIRED)
find_package(Monitoring REQUIRED)
find_package(Common REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(GLFW)
find_package(benchmark QUIET)
find_package(Arrow)
find_package(CURL REQUIRED)
find_package(OpenMP)

if (DDS_FOUND)
  add_definitions(-DENABLE_DDS)
  add_definitions(-DDDS_FOUND)
  set(OPTIONAL_DDS_LIBRARIES ${DDS_INTERCOM_LIBRARY_SHARED} ${DDS_PROTOCOL_LIBRARY_SHARED} ${DDS_USER_DEFAULTS_LIBRARY_SHARED})
  set(OPTIONAL_DDS_INCLUDE_DIR ${DDS_INCLUDE_DIR})
endif ()

set(CUDA_MINIMUM_VERSION "10.1")
if(DEFINED ENABLE_CUDA AND NOT ENABLE_CUDA)
  message(STATUS "CUDA explicitly disabled")
else()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
      set(CMAKE_CUDA_FLAGS "-Xptxas -O0 -Xcompiler -O0")
    else()
      set(CMAKE_CUDA_FLAGS "-Xptxas -O4 -Xcompiler -O4 -use_fast_math")
    endif()
    if(CUDA_GCCBIN)
      message(STATUS "Using as CUDA GCC version: ${CUDA_GCCBIN}")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-bindir ${CUDA_GCCBIN}")
    endif()
    enable_language(CUDA)
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT CUDA IN_LIST LANGUAGES)
      message(FATAL_ERROR "CUDA was found but cannot be enabled for some reason")
    endif()
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "${CUDA_MINIMUM_VERSION}")
      message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} found, but at least ${CUDA_MINIMUM_VERSION} required")
    endif()
    set(ENABLE_CUDA ON)
    if(CUDA_GCCBIN)
      #Ugly hack! Otherwise CUDA includes unwanted old GCC libraries leading to version conflicts
      set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "$ENV{CUDA_PATH}/lib64")
    endif()
    add_definitions(-DENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --compiler-options \"${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} -std=c++14\"")
  elseif(ENABLE_CUDA)
    message(FATAL_ERROR "CUDA explicitly enabled but could not be found")
  endif()
endif()

if (ENABLE_HIP)
  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
       set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
      set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")
  if(NOT DEFINED HCC_PATH)
    # Workaround to fix a potential FindHIP bug: find HCC_PATH ourselves
    set(_HCC_PATH "${HIP_PATH}/../hcc")
    get_filename_component(HCC_PATH ${_HCC_PATH} ABSOLUTE CACHE)
    unset(_HCC_PATH)
  endif()
  find_package(HIP REQUIRED)
  add_definitions(-DENABLE_HIP)
endif()

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
get_target_property(_boost_incdir Boost::boost INTERFACE_INCLUDE_DIRECTORIES)
if(FairMQInFairRoot_FOUND)
  # DEPRECATED: Remove this case, once we require FairMQ 1.2+
  get_target_property(_fairmq_incdir FairRoot::FairMQ INTERFACE_INCLUDE_DIRECTORIES)
  o2_define_bucket(NAME fairmq_bucket
    DEPENDENCIES FairRoot::FairMQ
    INCLUDE_DIRECTORIES ${_boost_incdir} ${_fairmq_incdir}
  )
else()
  get_target_property(_fairmq_incdir FairMQ::FairMQ INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(_fairlogger_incdir FairLogger::FairLogger INTERFACE_INCLUDE_DIRECTORIES)
  o2_define_bucket(NAME fairmq_bucket
    DEPENDENCIES FairMQ::FairMQ
    INCLUDE_DIRECTORIES ${_boost_incdir} ${_fairmq_incdir} ${_fairlogger_incdir}
  )
  set(_fairlogger_incdir)
endif()
set(_boost_incdir)
set(_fairmq_incdir)

o2_define_bucket(
  NAME
  glfw_bucket

  DEPENDENCIES
  O2FrameworkFoundation_bucket
  ${GLFW_LIBRARIES}

  INCLUDE_DIRECTORIES
  ${GLFW_INCLUDE_DIR}
)

o2_define_bucket(
  NAME
  headless_bucket

  DEPENDENCIES
  O2FrameworkFoundation_bucket
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
    arrow_bucket

    DEPENDENCIES
    common_boost_bucket
    ${ARROW_SHARED_LIB}

    SYSTEMINCLUDE_DIRECTORIES
    ${ARROW_INCLUDE_DIR}
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
    MemoryResources
    FairTools
    Headers
    fairmq_bucket
    AliceO2::Monitoring

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
)

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
    InfoLogger_bucket
    DEPENDENCIES
    ${InfoLogger_LIBRARIES}

    SYSTEMINCLUDE_DIRECTORIES
    ${InfoLogger_INCLUDE_DIRS}
)

o2_define_bucket(
    NAME
    O2FrameworkFoundation_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Framework/Foundation/include
)

o2_define_bucket(
    NAME
    O2FrameworkCore_bucket

    DEPENDENCIES
    arrow_bucket
    O2FrameworkFoundation_bucket
    O2DeviceApplication_bucket
    common_utils_bucket
    ROOTDataFrame
    ROOTVecOps
    Core
    Tree
    TreePlayer
    Net
    DebugGUI
    AliceO2::Monitoring
    AliceO2::Configuration
    InfoLogger_bucket
    AliceO2::Common
    CURL::libcurl

    SYSTEMINCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Utilities/PCG/include
)

o2_define_bucket(
    NAME
    O2FrameworkCore_benchmark_bucket

    DEPENDENCIES
    O2FrameworkCore_bucket
    $<IF:$<BOOL:${benchmark_FOUND}>,benchmark::benchmark,$<0:"">>
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
        DPLUtils_bucket

        DEPENDENCIES
        O2FrameworkCore_bucket
        Core
        Headers
        Framework
)

o2_define_bucket(
    NAME
    O2MessageMonitor_bucket

    DEPENDENCIES
    O2Device_bucket
    O2Device
)

# module DataFormats/Headers
o2_define_bucket(
    NAME
    data_format_headers_bucket

    DEPENDENCIES
    pmr_bucket
    Boost::container

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/Headers/include
    ${CMAKE_SOURCE_DIR}/DataFormats/MemoryResources/include
)

# module DataFormats/Detectors/TPC
o2_define_bucket(
    NAME
    data_format_TPC_bucket

    DEPENDENCIES
    data_format_headers_bucket
    data_format_reconstruction_bucket
    ReconstructionDataFormats
    Headers

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include
    ${CMAKE_SOURCE_DIR}/Algorithm/include
)

o2_define_bucket(
    NAME
    data_format_TOF_bucket

    DEPENDENCIES
    data_format_reconstruction_bucket
    ReconstructionDataFormats

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TOF/include
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
    ${CMAKE_SOURCE_DIR}/DataFormats/MemoryResources/include
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
    FairTools
    Headers
    fairmq_bucket

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
    fairmq_bucket
    Base FairTools Core MathCore Matrix Minuit Hist Geom GenVector RIO
    GPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/Constants/include
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
    DetectorsCommonDataFormats

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
    protobuf::libprotobuf
    Base
    FairTools
    ParBase
    ParMQ
    fairmq_bucket
    pthread Core Tree XMLParser Hist Net RIO z
    ${CURL_LIBRARIES}
    common_utils_bucket

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
    ${CMAKE_SOURCE_DIR}/Utilities/O2Device/include

    SYSTEMINCLUDE_DIRECTORIES
    ${PROTOBUF_INCLUDE_DIR}
    ${CURL_INCLUDE_DIRS}
)

o2_define_bucket(
    NAME
    root_base_bucket

    DEPENDENCIES
    Core RIO GenVector # ROOT

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
)

if(FairLogger_FOUND)
  # DEPRECATED: Remove this variable and use the value directly,
  # once we require FairMQ 1.2+
  set(FairLogger_DEP FairLogger::FairLogger)
endif()

# module DataFormats/MemoryResources
o2_define_bucket(
    NAME
    pmr_bucket

    DEPENDENCIES
    Boost::container
    fairmq_bucket
)

o2_define_bucket(
    NAME
    fairroot_geom

    DEPENDENCIES
    FairTools
    Base GeoBase ParBase Geom Core VMC Tree
    common_boost_bucket
    ${FairLogger_DEP}

    INCLUDE_DIRECTORIES
    ${FairLogger_INCDIR}
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
    fairmq_bucket
    common_boost_bucket
    Boost::thread
    Boost::serialization
    pthread
    MemoryResources
    ${FairLogger_DEP}

    INCLUDE_DIRECTORIES
    ${FairLogger_INCDIR}
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
    data_format_detectors_common_bucket
    DetectorsCommonDataFormats
    detectors_base_bucket
    DetectorsBase
    RIO
    SimConfig
    GPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/Common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include/
    ${CMAKE_SOURCE_DIR}/Common/SimConfig/include/

    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    steer_bucket

    DEPENDENCIES
    data_format_simulation_bucket
    SimulationDataFormat
    ITSMFTSimulation
    RIO
    Net
    SimConfig

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/Common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${MS_GSL_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}/fairmq
)


o2_define_bucket(
    NAME
    data_format_simulation_test_bucket

    DEPENDENCIES
    data_format_simulation_bucket
    SimulationDataFormat
)

o2_define_bucket(
    NAME
    data_format_reconstruction_bucket

    DEPENDENCIES
    fairroot_base_bucket
    root_physics_bucket
    data_format_detectors_common_bucket
    DetectorsCommonDataFormats
    CommonDataFormat
    GPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include/
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/Common/include/
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include/
    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    data_format_detectors_common_bucket

    DEPENDENCIES
    fairroot_base_bucket
    root_physics_bucket
    common_math_bucket
    data_format_headers_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/Common/include/
)

o2_define_bucket(
    NAME
    detectors_base_bucket

    DEPENDENCIES
    fairroot_base_bucket
    root_physics_bucket
    data_format_reconstruction_bucket
    common_utils_bucket
    GPUCommon_bucket
    O2GPUCommon
    ReconstructionDataFormats
    DataFormatsParameters
    CommonUtils
    Field
    fairmq_bucket
    Net
    VMC # ROOT
    Geom
    common_utils_bucket
    CommonUtils


    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Common/Field/include
    ${CMAKE_SOURCE_DIR}/Common/Constants/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Parameters/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
)

o2_define_bucket(
    NAME
    itsmft_base_bucket

    DEPENDENCIES
    fairroot_base_bucket
    configuration_bucket
    MathCore
    Geom
    RIO
    Hist
    ParBase
    Field
    SimulationDataFormat
    SimConfig
    CommonDataFormat
    detectors_base_bucket
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/Base/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
    ${CMAKE_SOURCE_DIR}/Common/SimConfig/include/
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
    Hist
    Graf
    Gpad
    Geom
    common_boost_bucket
    Boost::unit_test_framework
    ${FairLogger_DEP}
    RapidJSON

    INCLUDE_DIRECTORIES
    ${FairLogger_INCDIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${RAPIDJSON_INCLUDEDIR}/include
)

o2_define_bucket(
    NAME
    itsmft_simulation_bucket

    DEPENDENCIES
    itsmft_base_bucket
    data_format_itsmft_bucket
    configuration_bucket
    Graf
    Gpad
    DetectorsBase
    SimulationDataFormat
    ITSMFTBase
    DataFormatsITSMFT
    SimConfig

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Common/SimConfig/include
)

o2_define_bucket(
    NAME
    itsmft_reconstruction_bucket

    DEPENDENCIES
    itsmft_base_bucket
    data_format_itsmft_bucket
    common_utils_bucket
    #
    Graf
    Gpad
    DetectorsBase
    DataFormatsITSMFT
    ITSMFTBase
    CommonUtils

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/common/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
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
    itsmft_simulation_bucket
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
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
)

o2_define_bucket(
    NAME
    its_reconstruction_bucket

    DEPENDENCIES
    its_base_bucket
    data_format_itsmft_bucket
    data_format_its_bucket
    itsmft_reconstruction_bucket
    #
    ITSMFTBase
    ITSMFTReconstruction
    ITSBase
    DetectorsBase
    DataFormatsITS

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/ITS/include
)

o2_define_bucket(
    NAME
    its_tracking_bucket

    DEPENDENCIES
    data_format_its_bucket
    GPUCommon_bucket
    #
    DataFormatsITS
    DetectorsBase
    ITSBase

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/ITS/include
)

o2_define_bucket(
    NAME
    its_tracking_CUDA_bucket

    DEPENDENCIES
    #
    cuda
    cudart
    cudadevrt
    ITStracking

    INCLUDE_DIRECTORIES
    ${CUB_ROOT}
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/include
)

o2_define_bucket(
    NAME
    ITS_workflow_bucket

    DEPENDENCIES
    Framework
    its_reconstruction_bucket
    ITSReconstruction
    ITStracking

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/workflow/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/include
)

o2_define_bucket(
    NAME
    ITSMFT_workflow_bucket

    DEPENDENCIES
    Framework
    data_format_itsmft_bucket
    itsmft_reconstruction_bucket
    ITSMFTReconstruction
    DataFormatsITSMFT

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/workflow/include
)


o2_define_bucket(
    NAME
    fit_workflow_bucket

    DEPENDENCIES
    data_format_fit_bucket
    fit_reconstruction_bucket
    Framework
    T0Reconstruction
    DataFormatsFITT0
    DataFormatsFITV0
    
    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/workflow/include
)

o2_define_bucket(
    NAME
    GlobalTracking_workflow_bucket

    DEPENDENCIES
    Framework
    ReconstructionDataFormats
    GlobalTracking
    TPCWorkflow
    ITSWorkflow
    ITSMFTWorkflow
    FITWorkflow
    
    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/GlobalTrackingWorkflow/include
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
    ${CMAKE_SOURCE_DIR}/Common/Utils/include

    SYSTEMINCLUDE_DIRECTORIES
    ${Boost_INCLUDE_DIR}
    )

o2_define_bucket(
  NAME
  mergers_bucket

  DEPENDENCIES
  Base
  Headers
  Framework
  Core
  Hist
  arrow_bucket
  fairmq_bucket
  O2FrameworkCore_bucket

  INCLUDE_DIRECTORIES
  ${MS_GSL_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/DataFormats/MemoryResources/include
  ${CMAKE_SOURCE_DIR}/DataFormats/Headers/include
  ${CMAKE_SOURCE_DIR}/Framework/Core/include
  ${CMAKE_SOURCE_DIR}/Utilities/Mergers/include
)

o2_define_bucket(
    NAME
    tpc_base_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    common_vc_bucket
    common_math_bucket
    data_format_TPC_bucket
    ParBase
    MathUtils
    CCDB
    Core Hist Gpad
    SimulationDataFormat
    CommonDataFormat
    DataFormatsTPC

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/CCDB/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include
)

o2_define_bucket(
    NAME
    tpc_simulation_bucket

    DEPENDENCIES
    tpc_base_bucket
    data_format_TPC_bucket
    detectors_base_bucket
    TPCSpaceChargeBase_bucket
    Field
    DetectorsBase
    Generators
    TPCBase
    SimulationDataFormat
    DataFormatsTPC
    O2TPCSpaceChargeBase
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
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include
    ${CMAKE_SOURCE_DIR}/Steer/include
    ${MS_GSL_INCLUDE_DIR}
)


o2_define_bucket(
    NAME
    tpc_reconstruction_bucket

    DEPENDENCIES
    tpc_base_bucket
    data_format_TPC_bucket
    data_format_detectors_common_bucket
    TPCFastTransformation_bucket
    DetectorsCommonDataFormats
    DetectorsBase
    TPCBase
    DataFormatsTPC
    SimulationDataFormat
    CommonDataFormat
    ReconstructionDataFormats
    Geom
    MathCore
    RIO
    Hist
    DetectorsPassive
    Gen
    Base
    TreePlayer
    O2GPUTracking
    O2TPCFastTransformation
    TPCSimulation
    #the dependency on TPCSimulation should be removed at some point
    #perhaps 'Cluster' can be moved to base, or so

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/Passive/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/Common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Headers/include
    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    tpc_calibration_bucket

    DEPENDENCIES
    tpc_base_bucket
    data_format_TPC_bucket
    tpc_reconstruction_bucket
    DetectorsBase
    DataFormatsTPC
    TPCBase
    TPCReconstruction
    MathUtils

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include
)

o2_define_bucket(
    NAME
    tpc_monitor_bucket

    DEPENDENCIES
    DetectorsBase
    TPCBase
    TPCCalibration
    TPCReconstruction

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/calibration/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Headers/include
    ${Vc_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${FairLogger_INCDIR}
)

o2_define_bucket(
    NAME
    TPC_workflow_bucket

    DEPENDENCIES
    TPCReconstruction
    Framework
    DPLUtils

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Algorithm/include
   )

# base bucket for generators not needing any external stuff
o2_define_bucket(
    NAME
    generators_base_bucket

    DEPENDENCIES
    Base SimulationDataFormat MathCore RIO Tree
    fairroot_base_bucket
    # Gen is generator module from FairRoot
    Gen
    SimConfig

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/SimConfig/include
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
    common_utils_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include

)

o2_define_bucket(
    NAME
    mft_simulation_bucket

    DEPENDENCIES
    mft_base_bucket
    itsmft_simulation_bucket
    ITSMFTBase
    ITSMFTSimulation
    MFTBase
    DetectorsBase
    SimulationDataFormat
    common_utils_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/base/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
)

o2_define_bucket(
    NAME
    mft_reconstruction_bucket

    DEPENDENCIES
    mft_base_bucket
    itsmft_reconstruction_bucket
    data_format_mft_bucket
    ITSMFTBase
    ITSMFTReconstruction
    MFTBase
    MFTSimulation
    DetectorsBase
    DataFormatsITSMFT

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/common/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/MFT/simulation/include

)

o2_define_bucket(
    NAME
    tof_base_bucket

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
    CommonDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    trd_base_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    Geom
    Gpad
    Graf
    MathCore
    Matrix
    Physics
    ParBase
    VMC
    Geom
    SimulationDataFormat
    CommonDataFormat
    data_format_detectors_common_bucket
    DetectorsCommonDataFormats
    GPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
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
    CommonDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
)

o2_define_bucket(
    NAME
    passive_detector_bucket

    DEPENDENCIES
    fairroot_geom
    Field
    DetectorsBase
    SimConfig

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/Field/include
    ${CMAKE_SOURCE_DIR}/Detectors/Passive/include
    ${CMAKE_SOURCE_DIR}/Common/SimConfig/include
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
    detectors_base_bucket
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
    tof_base_bucket
    root_base_bucket
    detectors_base_bucket
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
    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    tof_reconstruction_bucket

    DEPENDENCIES
    tof_base_bucket
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
    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    tof_reconstruction_bucket

    DEPENDENCIES
    tof_base_bucket
    root_base_bucket
    data_format_TOF_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    TOFBase
    DetectorsBase
    SimulationDataFormat
    DataFormatsTOF

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TOF/base/include
#    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TOF/include
    ${MS_GSL_INCLUDE_DIR}
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
    CommonDataFormat
    detectors_base_bucket
    DetectorsBase
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/common/base/include

 )

o2_define_bucket(
    NAME
    fit_simulation_bucket

    DEPENDENCIES # library names
    data_format_fit_bucket
    fit_base_bucket
    root_base_bucket
    fairroot_geom
    DataFormatsFITT0
    DataFormatsFITV0
    RIO
    Graf
    Gpad
    Matrix
    Physics
    T0Base
    V0Base
    FDDBase
    DetectorsBase
    detectors_base_bucket
    SimulationDataFormat
    Core Hist # ROOT
    CommonDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/FIT/T0/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/FIT/V0/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/common/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/common/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/T0/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/V0/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/FDD/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/T0/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/V0/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/FDD/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)

o2_define_bucket(
    NAME
    hmpid_base_bucket

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
    CommonDataFormat
    CommonUtils

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Detectors/HMPID/base/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
)

o2_define_bucket(
    NAME
    zdc_base_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    Geom
    VMC
    SimulationDataFormat
    CommonDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Detectors/ZDC/base/include
)


o2_define_bucket(
    NAME
    fit_reconstruction_bucket

    DEPENDENCIES
    fit_base_bucket
    data_format_fit_bucket
    T0Base
    V0Base
    FDDBase
    DataFormatsFITT0
    DataFormatsFITV0
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/FIT/T0/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/FIT/V0/include
    ${CMAKE_SOURCE_DIR}/Detectors/FIT/T0/reconstruction/include
)

o2_define_bucket(
    NAME
    data_format_fit_bucket

    DEPENDENCIES
    fit_base_bucket
    T0Base
    V0Base
    FDDBase
    DetectorsBase

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/FIT/include
)

o2_define_bucket(
    NAME
    hmpid_simulation_bucket

    DEPENDENCIES # library names
    hmpid_base_bucket
    HMPIDBase
    root_base_bucket
    detectors_base_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    DetectorsBase
    SimulationDataFormat
    Core Hist # ROOT

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/HMPID/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
)


o2_define_bucket(
    NAME
    zdc_simulation_bucket

    DEPENDENCIES # library names
    zdc_base_bucket
    ZDCBase
    detectors_base_bucket
    fairroot_geom
    RIO
    DetectorsBase
    SimulationDataFormat
    Core

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ZDC/simulation/include
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
    data_format_simulation_bucket
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
)

o2_define_bucket(
    NAME
    phos_simulation_bucket

    DEPENDENCIES
    phos_base_bucket
    root_base_bucket
    fairroot_geom
    detectors_base_bucket
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
    phos_reconstruction_bucket

    DEPENDENCIES
    phos_base_bucket
    phos_simulation_bucket
    root_base_bucket
    PHOSBase
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics


    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/PHOS/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/PHOS/reconstruction/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include

)

o2_define_bucket(
    NAME
    cpv_base_bucket

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
    data_format_simulation_bucket
    SimulationDataFormat

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
)

o2_define_bucket(
    NAME
    cpv_simulation_bucket

    DEPENDENCIES
    cpv_base_bucket
    root_base_bucket
    fairroot_geom
    detectors_base_bucket
    RIO
    Graf
    Gpad
    Matrix
    Physics
    CPVBase
    DetectorsBase
    SimulationDataFormat


    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Detectors/CPV/base/include
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
    spacepoint_calib_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    MathCore
    Matrix
    tpc_base_bucket
    common_utils_bucket
    common_math_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/calibration/SpacePoints/include
)

o2_define_bucket(
    NAME
    trd_simulation_bucket

    DEPENDENCIES
    trd_base_bucket
    root_base_bucket
    fairroot_geom
    RIO
    Graf
    Gpad
    Matrix
    Physics
    TRDBase
    DetectorsBase
    detectors_base_bucket
    SimulationDataFormat
    common_utils_bucket
    CommonUtils

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

    #-- precise modules follow
    SimConfig
    SimSetup
    DetectorsPassive
    TPCSimulation
    TPCReconstruction
    ITSSimulation
    MFTSimulation
    MCHSimulation
    MIDSimulation
    TRDSimulation
    EMCALSimulation
    TOFSimulation
    T0Simulation
    V0Simulation
    FDDSimulation
    HMPIDSimulation
    PHOSSimulation
    CPVSimulation
    PHOSReconstruction
    ZDCSimulation
    Field
    Generators
    DataFormatsParameters
    Framework
)

# a bucket for "global" executables/macros
o2_define_bucket(
    NAME
    digitizer_workflow_bucket

    DEPENDENCIES
    #-- buckets follow
    fairroot_base_bucket
    fit_simulation_bucket
    #-- precise modules follow
    Steer
    Framework
    DetectorsCommonDataFormats
    CommonDataFormat
    TPCSimulation
    TPCWorkflow
    DataFormatsTPC
    ITSSimulation
    MFTSimulation
    ITSMFTBase
    TOFSimulation
    TOFReconstruction
    FITSimulation
    T0Simulation
    FDDSimulation
    EMCALSimulation
    HMPIDBase
    HMPIDSimulation
    MCHBase
    MCHSimulation
    TRDBase
    TRDSimulation
    MIDSimulation
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
    ${CMAKE_SOURCE_DIR}/CCDB/include
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
    common_boost_bucket

    INCLUDE_DIRECTORIES
)

o2_define_bucket(
  NAME
  data_parameters_bucket

  DEPENDENCIES
  Core
  data_format_detectors_common_bucket
  DetectorsCommonDataFormats

  INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/Detectors/Base/include
  ${CMAKE_SOURCE_DIR}/Common/Constants/include
  ${CMAKE_SOURCE_DIR}/Common/Types/include
  ${CMAKE_SOURCE_DIR}/Detectors/Common/include/DetectorsCommonDataFormats
)

o2_define_bucket(
  NAME
  common_utils_bucket

  DEPENDENCIES
  Core Tree
  ReconstructionDataFormats # for test dependency only
  common_boost_bucket
  Boost::iostreams
  DataFormatsMID

  INCLUDE_DIRECTORIES
  ${ROOT_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/Common/Utils/include
  ${CMAKE_SOURCE_DIR}/include/ReconstructionDataFormats # for test dependency only
)

o2_define_bucket(
    NAME
    data_format_common_bucket

    DEPENDENCIES
    fairroot_base_bucket
    Core RIO

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/Common/Constants/include
)

o2_define_bucket(
    NAME
    mch_base_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
)

o2_define_bucket(
    NAME
    mch_simulation_bucket

    DEPENDENCIES
    root_base_bucket
    fairroot_base_bucket
    DetectorsBase
    detectors_base_bucket
    SimulationDataFormat
    RapidJSON
    mch_mapping_interface_bucket
    mch_mapping_impl3_bucket
    MCHMappingImpl3

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${RAPIDJSON_INCLUDEDIR}/include
    ${MS_GSL_INCLUDE_DIR}
)

o2_define_bucket(
    NAME
    mch_simulation_test_bucket

    DEPENDENCIES
    mch_simulation_bucket
    mch_mapping_impl3_bucket
    MCHMappingImpl3
    MCHSimulation
)

o2_define_bucket(
    NAME
    mch_preclustering_bucket

    DEPENDENCIES
    fairroot_base_bucket
    MCHBase
    Framework

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/MUON/MCH/Base/include
)

o2_define_bucket(
    NAME
    data_format_itsmft_bucket

    DEPENDENCIES
    data_format_reconstruction_bucket
    #
    ReconstructionDataFormats

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include
)

o2_define_bucket(
    NAME
    data_format_its_bucket

    DEPENDENCIES
    data_format_reconstruction_bucket
    #
    ReconstructionDataFormats

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/ITS/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include
    ${CMAKE_SOURCE_DIR}/Common/Constants/include
)

o2_define_bucket(
    NAME
    data_format_mft_bucket

    DEPENDENCIES
    data_format_reconstruction_bucket
    #
    ReconstructionDataFormats

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/MFT/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include
    ${CMAKE_SOURCE_DIR}/Common/Constants/include
)

o2_define_bucket(
    NAME
    global_tracking_bucket

    DEPENDENCIES
    data_format_simulation_bucket
    data_format_reconstruction_bucket
    data_format_common_bucket
    data_format_TPC_bucket
    data_format_TOF_bucket
    data_format_fit_bucket
    its_reconstruction_bucket
    data_format_itsmft_bucket
    common_field_bucket
    detectors_base_bucket
    its_base_bucket
    tpc_base_bucket
    tpc_reconstruction_bucket
    tof_base_bucket
    GPUTracking_bucket
    data_parameters_bucket
    common_utils_bucket
    common_math_bucket
    #
    SimulationDataFormat
    ReconstructionDataFormats
    CommonDataFormat
    ITSReconstruction
    TPCReconstruction
    DataFormatsITSMFT
    DataFormatsFITT0
    DetectorsBase
    DataFormatsTPC
    DataFormatsTOF
    DataFormatsParameters
    ITSBase
    TPCBase
    TOFBase
    CommonUtils
    MathUtils
    Field
    O2GPUTracking
    O2TPCFastTransformation
    RIO
    Core
    Geom

    INCLUDE_DIRECTORIES
    ${FAIRROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/include
    ${CMAKE_SOURCE_DIR}/DataFormats/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/simulation/include
    ${CMAKE_SOURCE_DIR}/Common/Field/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/ITSMFT/common/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/FIT/T0/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TPC/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/TOF/base/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Common/Utils/include
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/Common/Constants/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Parameters/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Interface
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/SliceTracker
)

o2_define_bucket(
  NAME
  mch_contour_bucket

  INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/Detectors/MUON/MCH/Contour/include
)

o2_define_bucket(
  NAME
  mch_mapping_interface_bucket

  INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/Detectors/MUON/MCH/Mapping/Interface/include
)

o2_define_bucket(
  NAME
  mch_mapping_impl3_bucket

  DEPENDENCIES
  mch_mapping_interface_bucket

  INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/Detectors/MUON/MCH/Mapping/Impl3/src
  ${CMAKE_BINARY_DIR}/Detectors/MUON/MCH/Mapping/Impl3 # for the mchmappingimpl3_export.h generated file

  SYSTEMINCLUDE_DIRECTORIES
  ${Boost_INCLUDE_DIR}
)

o2_define_bucket(
  NAME
  mch_mapping_segcontour_bucket

  DEPENDENCIES
  mch_contour_bucket
  mch_mapping_impl3_bucket
  Boost::program_options
  MCHMappingImpl3

  INCLUDE_DIRECTORIES
  ${CMAKE_SOURCE_DIR}/Detectors/MUON/MCH/Mapping/SegContour/include

  SYSTEMINCLUDE_DIRECTORIES
  ${Boost_INCLUDE_DIR}
)

o2_define_bucket(
  NAME
  mch_mapping_test_bucket

  DEPENDENCIES
  $<IF:$<BOOL:${benchmark_FOUND}>,benchmark::benchmark,$<0:"">>
  mch_mapping_segcontour_bucket
  MCHMappingSegContour3
  RapidJSON

  INCLUDE_DIRECTORIES
  ${RAPIDJSON_INCLUDEDIR}/include
)

o2_define_bucket(
    NAME
    data_format_mid_bucket

    DEPENDENCIES
    Boost::serialization
    common_math_bucket

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/MathUtils/include
    ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/MUON/MID/include
)

o2_define_bucket(
    NAME
    mid_base_bucket

    DEPENDENCIES
    data_format_mid_bucket
    DataFormatsMID

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/MUON/MID/Base/include
)

o2_define_bucket(
    NAME
    mid_base_test_bucket

    DEPENDENCIES
    Boost::unit_test_framework
    RapidJSON
    mid_base_bucket
    MIDBase
)

o2_define_bucket(
    NAME
    mid_clustering_bucket

    DEPENDENCIES
    fairroot_base_bucket
    MIDBase
)

o2_define_bucket(
    NAME
    mid_clustering_test_bucket

    DEPENDENCIES
    Boost::unit_test_framework
    $<IF:$<BOOL:${benchmark_FOUND}>,benchmark::benchmark,$<0:"">>
    mid_clustering_bucket
    MIDClustering
)


o2_define_bucket(
    NAME
    mid_simulation_test_bucket

    DEPENDENCIES
    Boost::unit_test_framework
    $<IF:$<BOOL:${benchmark_FOUND}>,benchmark::benchmark,$<0:"">>
    RapidJSON
    MIDBase
    MIDSimulation
    MIDClustering

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/MUON/MID/Simulation/src
    ${CMAKE_SOURCE_DIR}/Detectors/MUON/MID/Clustering/src
)

o2_define_bucket(
    NAME
    mid_testingSimTools_bucket

    DEPENDENCIES
    MIDBase
)

o2_define_bucket(
    NAME
    mid_tracking_bucket

    DEPENDENCIES
    fairroot_base_bucket
    MIDBase
)

o2_define_bucket(
    NAME
    mid_tracking_test_bucket

    DEPENDENCIES
    Boost::unit_test_framework
    $<IF:$<BOOL:${benchmark_FOUND}>,benchmark::benchmark,$<0:"">>
    mid_tracking_bucket
    mid_testingSimTools_bucket
    MIDTracking
    MIDTestingSimTools

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/MUON/MID/TestingSimTools/include
)


o2_define_bucket(
    NAME
    simulation_setup_bucket

    DEPENDENCIES
    ${Geant3_LIBRARIES}
    ${Geant4_LIBRARIES}
    ${Geant4VMC_LIBRARIES}
    ${VGM_LIBRARIES}
    fairroot_geom
    SimulationDataFormat
    DetectorsPassive
    pythia6 # this is needed by Geant3 and EGPythia6
    EGPythia6 # this is needed by Geant4 (TPythia6Decayer)

    INCLUDE_DIRECTORIES
    ${Geant4VMC_INCLUDE_DIRS}
    ${Geant4_INCLUDE_DIRS}
    ${Geant3_INCLUDE_DIRS}
    ${FAIRROOT_INCLUDE_DIR}
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Common/SimConfig/include
)

o2_define_bucket(
    NAME
    utility_datacompression_bucket

    DEPENDENCIES
    CommonUtils
    common_boost_bucket

    INCLUDE_DIRECTORIES
)

o2_define_bucket(
    NAME
    mid_simulation_bucket

    DEPENDENCIES
    data_format_simulation_bucket
    root_base_bucket
    mid_base_bucket
    MIDBase
    SimulationDataFormat
)

o2_define_bucket(
    NAME
    mch_tracking_bucket

    DEPENDENCIES
    fairroot_base_bucket
    Field
    MCHBase
    Framework

    INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Common/Field/include
    ${CMAKE_SOURCE_DIR}/Detectors/MUON/MCH/Base/include
)

o2_define_bucket(
    NAME
    GPUCommon_bucket

    DEPENDENCIES
    Core

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/GPU/Common
)

o2_define_bucket(
    NAME
    TPCFastTransformation_bucket

    DEPENDENCIES
    dl
    pthread
    root_base_bucket
    common_vc_bucket
    GPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/GPU/TPCFastTransformation
)

o2_define_bucket(
    NAME
    GPUTracking_bucket

    DEPENDENCIES
    dl
    pthread
    root_base_bucket
    common_vc_bucket
    TRDBase
    ITStracking
    GPUCommon_bucket
    TPCFastTransformation_bucket
    O2TPCFastTransformation
    data_format_TPC_bucket
    Gpad
    RIO
    Graf
    glfw_bucket
    DebugGUI

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Global
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/SliceTracker
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TRDTracking
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Interface
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/HLTHeaders
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Standalone
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/ITS
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/dEdx
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TPCConvert
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Standalone/display
    ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Standalone/qa
    ${CMAKE_SOURCE_DIR}/Framework/Core/include
    ${CMAKE_SOURCE_DIR}/Detectors/Base/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/include
    ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/include
)

o2_define_bucket(
    NAME
    GPUTrackingHIP_bucket

    DEPENDENCIES
    GPUTracking_bucket
)

o2_define_bucket(
    NAME
    GPUTrackingCUDA_bucket

    DEPENDENCIES
    GPUTracking_bucket
    ITStrackingCUDA
)

o2_define_bucket(
    NAME
    GPUTrackingOCL_bucket

    DEPENDENCIES
    GPUTracking_bucket
)

o2_define_bucket(
    NAME
    TPCSpaceChargeBase_bucket

    DEPENDENCIES
    root_base_bucket Hist MathCore Matrix Physics GPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/GPU/TPCSpaceChargeBase
)
