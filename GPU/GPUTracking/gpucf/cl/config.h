#if !defined(CONFIG_H)
#    define  CONFIG_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


#include "shared/Digit.h"
#include "shared/tpc.h"


typedef PackedDigit Digit;


#if defined(CHARGEMAP_TYPE_HALF)
  typedef half charge_t;
#else
  typedef float charge_t;
#endif


#if defined(UNROLL_LOOPS)
  #define LOOP_UNROLL_ATTR __attribute__((opencl_unroll_hint))
#else
  #define LOOP_UNROLL_ATTR __attribute__((opencl_unroll_hint(1)))
#endif


inline size_t chargemapIdx(global_pad_t gpad, timestamp time)
{
    time += PADDING;

#if defined(CHARGEMAP_4x4_TILING_LAYOUT) \
    || defined(CHARGEMAP_4x8_TILING_LAYOUT) \
    || defined(CHARGEMAP_8x4_TILING_LAYOUT)

#define CHARGEMAP_TILING_LAYOUT

#if defined(CHARGEMAP_4x4_TILING_LAYOUT)
  #define TILE_WIDTH 4
  #define TILE_HEIGHT 4
#elif defined(CHARGEMAP_4x8_TILING_LAYOUT)
  #define TILE_WIDTH 4
  #define TILE_HEIGHT 8
#elif defined(CHARGEMAP_8x4_TILING_LAYOUT)
  #define TILE_WIDTH 8
  #define TILE_HEIGHT 4
#else
  #error("Unknown tiling layout")
#endif

    const size_t tileW = TILE_WIDTH;
    const size_t tileH = TILE_HEIGHT;
    const size_t widthInTiles = (TPC_NUM_OF_PADS + tileW - 1) / tileW;

    const size_t tilePad  = gpad / tileW;
    const size_t tileTime = time / tileH;

    const size_t inTilePad = gpad % tileW;
    const size_t inTileTime = time % tileH;

    return (tileTime * widthInTiles + tilePad) * (tileW * tileH)
        + inTileTime * tileW + inTilePad;

#undef TILE_WIDTH
#undef TILE_HEIGHT

#elif defined(CHARGEMAP_PAD_MAJOR_LAYOUT)

    return TPC_MAX_TIME_PADDED * gpad + time;

#else // Use row time-major layout

    return TPC_NUM_OF_PADS * time + gpad;

#endif
}


#define ACCESS_2D(map, idxFunc, gpad, time) map[idxFunc(gpad, time)]

#define CHARGE(map, gpad, time) ACCESS_2D(map, chargemapIdx, gpad, time)
#define IS_PEAK(map, gpad, time) ACCESS_2D(map, chargemapIdx, gpad, time)
#define PEAK_COUNT(map, gpad, time) ACCESS_2D(map, chargemapIdx, gpad, time)

#endif //!defined(CONFIG_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
