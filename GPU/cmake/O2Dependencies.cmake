o2_define_bucket(
    NAME
    AliTPCCommonBase_bucket

    DEPENDENCIES

    INCLUDE_DIRECTORIES
    ${ALITPCCOMMON_DIR}/sources/Common
)

o2_define_bucket(
    NAME
    TPCFastTransformation_bucket

    DEPENDENCIES
    dl
    pthread
    root_base_bucket
    common_vc_bucket
    AliTPCCommonBase_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${ALITPCCOMMON_DIR}/sources/TPCFastTransformation
)

o2_define_bucket(
    NAME
    CAGPUTracking_bucket

    DEPENDENCIES
    dl
    pthread
    root_base_bucket
    common_vc_bucket
    TRDBase
    ITStracking
    AliTPCCommonBase_bucket
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
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/GlobalTracker
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/SliceTracker
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/Merger
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/TRDTracking
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/Interface
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/HLTHeaders
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/Standalone
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/Standalone/cmodules
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/Standalone/display
    ${ALITPCCOMMON_DIR}/sources/CAGPUTracking/Standalone/qa
    ${CMAKE_SOURCE_DIR}/Framework/Core/include
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/include
    ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/include
)

o2_define_bucket(
    NAME
    CAGPUTrackingCUDA_bucket

    DEPENDENCIES
    CAGPUTracking_bucket
    ITStrackingCUDA
)

o2_define_bucket(
    NAME
    CAGPUTrackingOCL_bucket

    DEPENDENCIES
    CAGPUTracking_bucket
)

o2_define_bucket(
    NAME
    TPCSpaceChargeBase_bucket

    DEPENDENCIES
    root_base_bucket Hist MathCore Matrix Physics AliTPCCommonBase_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${ALITPCCOMMON_DIR}/sources/TPCSpaceChargeBase
)
