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


#if defined(CHARGEMAP_SQUARE_CACHE_LINES)

  #if !defined(CACHE_LINE_HEIGHT)
    #error("Cache line height must be specified.")
  #endif

  #if !defined(CACHE_LINE_WIDTH)
    #error("Cache line width must be specified.")
  #endif

  #define IDX_OF_CACHE_LINE(row, pad, time) \
      ((TPC_NUM_OF_PADS / CACHE_LINE_WIDTH) * ((time) / CACHE_LINE_HEIGHT) \
        + (TPC_GLOBAL_PAD_IDX((row), (pad)) * CACHE_LINE_WIDTH) / TPC_NUM_OF_PADS)

  #define IDX_IN_CACHE_LINE(row, pad, time) \
      (CACHE_LINE_WIDTH * ((time) % CACHE_LINE_HEIGHT) \
      + TPC_GLOBAL_PAD_IDX((row), (pad)) % CACHE_LINE_WIDTH)

  #define CHARGEMAP_IDX_IMPL(row, pad, time) \
      (IDX_OF_CACHE_LINE((row), (pad), (time)) \
      + IDX_IN_CACHE_LINE((row), (pad), (time)))

#else

  #define CHARGEMAP_IDX_IMPL(row, pad, time) (TPC_NUM_OF_PADS * (time) \
        + TPC_GLOBAL_PAD_IDX(row, pad))

#endif


#define CHARGEMAP_IDX(row, pad, time) \
      CHARGEMAP_IDX_IMPL(row, (pad)+PADDING, (time)+PADDING)

#define CHARGE(map, row, pad, time) map[CHARGEMAP_IDX(row, pad, time)]

#define DIGIT_CHARGE(map, digit) CHARGE(map, digit.row, digit.pad, digit.time)

#endif //!defined(CONFIG_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
