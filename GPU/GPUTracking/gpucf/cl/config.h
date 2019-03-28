#if !defined(CONFIG_H)
#    define  CONFIG_H

#pragma OPENCL cl_khr_fp16 : require


#include "shared/Cluster.h"
#include "shared/Digit.h"
#include "shared/tpc.h"


typedef FloatCluster Cluster;

#if defined(USE_PACKED_DIGIT)
  typedef PackedDigit Digit;
#else
  typedef PaddedDigit Digit;
#endif


inline size_t chargemapIdx(uchar row, uchar pad, short time)
{
#if defined(CHARGEMAP_TILING_LAYOUT)

    const int tileW = 4;
    const int tileH = 4;
    const int widthInTiles = (TPC_NUM_OF_PADS + tileW - 1) / tileW;

    const size_t globPad = tpcGlobalPadIdx(row, pad);

    size_t tilePad  = globPad / tileW;
    size_t tileTime = time / tileH;

    size_t inTilePad = globPad % tileW;
    size_t inTileTime = time % tileH;

    return (tileTime * widthInTiles + tilePad) * (tileW * tileH)
        + inTileTime * tileW + inTilePad;

#else // Use row layout

      return TPC_NUM_OF_PADS * time + tpcGlobalPadIdx(row, pad);

#endif
}


#if defined(CHARGEMAP_IDX_MACRO)
  #define CHARGEMAP_IDX(row, pad, time) \
      (TPC_NUM_OF_PADS * (time + PADDING) + TPC_PADS_PER_ROW_PADDED * row + pad + PADDING)
#else
  #define CHARGEMAP_IDX(row, pad, time) \
            chargemapIdx(row, (pad)+PADDING, (time)+PADDING)
#endif

#define CHARGE(map, row, pad, time) map[CHARGEMAP_IDX(row, pad, time)]

#define DIGIT_CHARGE(map, digit) CHARGE(map, digit.row, digit.pad, digit.time)

#endif //!defined(CONFIG_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
