#if !defined(CONFIG_H)
#    define  CONFIG_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


#include "shared/Cluster.h"
#include "shared/Digit.h"
#include "shared/tpc.h"


typedef FloatCluster Cluster;

#if defined(USE_PACKED_DIGIT)
  typedef PackedDigit Digit;
#else
  typedef PaddedDigit Digit;
#endif


#if defined(CHARGEMAP_TYPE_HALF)
  typedef half charge_t;
#else
  typedef float charge_t;
#endif


inline size_t chargemapIdx(uchar row, uchar pad, ushort time)
{
#if defined(CHARGEMAP_4x4_TILING_LAYOUT) \
    || defined(CHARGEMAP_4x8_TILING_LAYOUT) \
    || defined(CHARGEMAP_8x4_TILING_LAYOUT)

#if defined(CHARGEMAP_4x4_TILING_LAYOUT)
  #define TILE_WIDTH 4
  #define TILE_HEIGHT 4
#elif defined(CHARGEMAP_4x8_TILING_LAYOUT)
  #define TILE_WIDTH 4
  #define TILE_HEIGHT 8
#elif defined(CHARGEMAP_8x4_TILING_LAYOUT)
  #define TILE_WIDTH 8
  #define TILE_HEIGHT 4
#endif

    const size_t tileW = TILE_WIDTH;
    const size_t tileH = TILE_HEIGHT;
    const size_t widthInTiles = (TPC_NUM_OF_PADS + tileW - 1) / tileW;

    const size_t globPad = tpcGlobalPadIdx(row, pad);

    size_t tilePad  = globPad / tileW;
    size_t tileTime = time / tileH;

    size_t inTilePad = globPad % tileW;
    size_t inTileTime = time % tileH;

    return (tileTime * widthInTiles + tilePad) * (tileW * tileH)
        + inTileTime * tileW + inTilePad;

#undef TILE_WIDTH
#undef TILE_HEIGHT

#elif defined(CHARGEMAP_PAD_MAJOR_LAYOUT)

    return TPC_MAX_TIME_PADDED * tpcGlobalPadIdx(row, pad) + time;

#else // Use row time-major layout

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
