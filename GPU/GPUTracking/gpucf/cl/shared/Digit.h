#if !defined(SHARED_DIGIT_H)
#    define  SHARED_DIGIT_H

#include "tpc.h"


typedef struct PaddedDigit_s
{
    SHARED_FLOAT charge;
    timestamp time;
    pad_t pad;
    row_t row;
    cru_t cru;
    /*SHARED_UCHAR padding[3]; Implicit padding to keep struct 4 byte aligned*/
} PaddedDigit;

#define PADDED_DIGIT_SIZE 12


typedef struct PackedDigit_s
{
    SHARED_FLOAT charge;
    timestamp time;
    pad_t pad;
    row_t row;
} PackedDigit;

#define PACKED_DIGIT_SIZE 8


#endif //!defined(SHARED_DIGIT_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
