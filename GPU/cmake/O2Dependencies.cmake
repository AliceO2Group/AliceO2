o2_define_bucket(
    NAME
    AliGPUCommon_bucket

    DEPENDENCIES

    INCLUDE_DIRECTORIES
    ${ALIGPU_DIR}/sources/Common
)

o2_define_bucket(
    NAME
    TPCFastTransformation_bucket

    DEPENDENCIES
    dl
    pthread
    root_base_bucket
    common_vc_bucket
    AliGPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${ALIGPU_DIR}/sources/TPCFastTransformation
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
    AliGPUCommon_bucket
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
    ${ALIGPU_DIR}/sources/GPUTracking/Global
    ${ALIGPU_DIR}/sources/GPUTracking/Base
    ${ALIGPU_DIR}/sources/GPUTracking/SliceTracker
    ${ALIGPU_DIR}/sources/GPUTracking/Merger
    ${ALIGPU_DIR}/sources/GPUTracking/TRDTracking
    ${ALIGPU_DIR}/sources/GPUTracking/Interface
    ${ALIGPU_DIR}/sources/GPUTracking/HLTHeaders
    ${ALIGPU_DIR}/sources/GPUTracking/Standalone
    ${ALIGPU_DIR}/sources/GPUTracking/
    ${ALIGPU_DIR}/sources/GPUTracking/Standalone/display
    ${ALIGPU_DIR}/sources/GPUTracking/Standalone/qa
    ${CMAKE_SOURCE_DIR}/Framework/Core/include
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
    root_base_bucket Hist MathCore Matrix Physics AliGPUCommon_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${ALIGPU_DIR}/sources/TPCSpaceChargeBase
)
