#if !defined(CONFIG_H)
#    define  CONFIG_H

#pragma OPENCL cl_khr_fp16 : require


#include "shared/Cluster.h"
#include "shared/Digit.h"


#if defined(USE_HALF_TYPES)
typedef HalfCluster Cluster;
typedef HalfDigit   Digit;
typedef half        float_t;
#else
typedef FloatCluster Cluster;
typedef FloatDigit   Digit;
typedef float        float_t;
#endif

#endif //!defined(CONFIG_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
