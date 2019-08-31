#if !defined(SHARED_TPC_H)
#    define  SHARED_TPC_H

#include "types.h"

#define PADDING_PAD 2
#define PADDING_TIME 2
#define TPC_SECTORS 36
#define TPC_ROWS_PER_CRU 18
#define TPC_NUM_OF_ROWS 152
#define TPC_PADS_PER_ROW 138
#define TPC_PADS_PER_ROW_PADDED (TPC_PADS_PER_ROW+PADDING_PAD)
#define TPC_NUM_OF_PADS (TPC_NUM_OF_ROWS * TPC_PADS_PER_ROW_PADDED + PADDING_PAD)
#define TPC_MAX_TIME 4000
#define TPC_MAX_TIME_PADDED (TPC_MAX_TIME+2*PADDING_TIME)


typedef SHARED_USHORT timestamp;
typedef SHARED_UCHAR pad_t;
typedef SHARED_USHORT global_pad_t;
typedef SHARED_UCHAR row_t;
typedef SHARED_UCHAR cru_t;


// Maps the position of a pad given as row and index in that row to a unique
// index between 0 and TPC_NUM_OF_PADS.
inline global_pad_t tpcGlobalPadIdx(row_t row, pad_t pad)
{
    return TPC_PADS_PER_ROW_PADDED * row + pad + PADDING_PAD;
}

#endif //!defined(SHARED_TPC_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
