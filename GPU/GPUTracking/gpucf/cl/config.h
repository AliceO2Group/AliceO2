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


size_t idxSquareTiling(global_pad_t gpad, timestamp time, size_t N)
{
    /* time += PADDING; */ 

    /* const size_t C = TPC_NUM_OF_PADS + N - 1; */

    /* const size_t inTilePad = gpad % N; */
    /* const size_t inTileTime = time % N; */

    /* return N * (time * C + gpad + inTileTime) + inTilePad; */

    time += PADDING;

    const size_t tileW = N;
    const size_t tileH = N;
    const size_t widthInTiles = (TPC_NUM_OF_PADS + tileW - 1) / tileW;

    const size_t tilePad  = gpad / tileW;
    const size_t tileTime = time / tileH;

    const size_t inTilePad = gpad % tileW;
    const size_t inTileTime = time % tileH;

    return (tileTime * widthInTiles + tilePad) * (tileW * tileH)
        + inTileTime * tileW + inTilePad;
}

size_t idxTiling4x4(global_pad_t gpad, timestamp time)
{
    return idxSquareTiling(gpad, time, 4);
}

size_t idxTiling8x8(global_pad_t gpad, timestamp time)
{
    return idxSquareTiling(gpad, time, 8);
}



inline size_t chargemapIdx(global_pad_t gpad, timestamp time)
{

#if defined(CHARGEMAP_4x4_TILING_LAYOUT) \
    || defined(CHARGEMAP_4x8_TILING_LAYOUT) \
    || defined(CHARGEMAP_8x4_TILING_LAYOUT)
#define CHARGEMAP_TILING_LAYOUT
#endif

#if defined(CHARGEMAP_4x4_TILING_LAYOUT)
    
    return idxTiling4x4(gpad, time);

#elif defined(CHARGEMAP_4x8_TILING_LAYOUT)
  #define TILE_WIDTH 4
  #define TILE_HEIGHT 8

    time += PADDING;

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

#elif defined(CHARGEMAP_8x4_TILING_LAYOUT)
  #define TILE_WIDTH 8
  #define TILE_HEIGHT 4

    time += PADDING;

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

    time += PADDING;
    return TPC_MAX_TIME_PADDED * gpad + time;

#else // Use row time-major layout

    time += PADDING;
    return TPC_NUM_OF_PADS * time + gpad;

#endif
}


size_t safeIdx(global_pad_t gpad, timestamp time)
{
    size_t allElements = TPC_MAX_TIME_PADDED * TPC_NUM_OF_PADS;

    size_t id = get_global_linear_id();

    if (gpad >= TPC_NUM_OF_PADS)
    {
        printf("%lu: gpad = %hu\n", id, gpad);
        return chargemapIdx(0, 0);
    }

    if (time+PADDING >= TPC_MAX_TIME_PADDED)
    {
        printf("%lu: time = %hu\n", id, time);
        return chargemapIdx(0, 0);
    }

    size_t ind = idxTiling8x8(gpad, time);

    if (ind >= allElements)
    {
        printf("%lu: gpad=%hu, time=%hu, elems = %lu, ind=%lu\n", 
                id, gpad, time, allElements, ind);
        return chargemapIdx(0,0);
    }

    return ind;
}



#define ACCESS_2D(map, idxFunc, gpad, time) map[idxFunc(gpad, time)]

#define CHARGE(map, gpad, time) ACCESS_2D(map, chargemapIdx, gpad, time)

#if defined(CHARGEMAP_TILING_LAYOUT)
/* #if 0 */
# define IS_PEAK(map, gpad, time) ACCESS_2D(map, idxTiling8x8, gpad, time)
#else
# define IS_PEAK(map, gpad, time) ACCESS_2D(map, chargemapIdx, gpad, time)
#endif

#endif //!defined(CONFIG_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
