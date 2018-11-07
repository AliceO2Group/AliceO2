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
    TPCCAGPUTracking_bucket

    DEPENDENCIES
    dl
    pthread
    root_base_bucket
    common_vc_bucket
    TRDBase
    ITStracking
    AliTPCCommonBase_bucket

    INCLUDE_DIRECTORIES
    ${ROOT_INCLUDE_DIR}
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/GlobalTracker
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/SliceTracker
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/Merger
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/TRDTracking
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/Interface
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/HLTHeaders
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/Standalone/cmodules
    ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking/Standalone/include
)

o2_define_bucket(
    NAME
    TPCCAGPUTrackingCUDA_bucket

    DEPENDENCIES
    TPCCAGPUTracking_bucket
    ITStrackingCUDA
)

o2_define_bucket(
    NAME
    TPCCAGPUTrackingOCL_bucket

    DEPENDENCIES
    TPCCAGPUTracking_bucket
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
