#if !defined(CONFIG_H)
#    define  CONFIG_H

#pragma OPENCL cl_khr_fp16 : require


#include "shared/Cluster.h"
#include "shared/Digit.h"


typedef FloatCluster Cluster;

#if defined(USE_PACKED_DIGIT)
typedef PackedDigit Digit;
#else
typedef PaddedDigit Digit;
#endif

#endif //!defined(CONFIG_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
