#if !defined(SHARED_DIGIT_H)
#    define  SHARED_DIGIT_H

#include "types.h"


typedef struct PaddedDigit_s
{
    SHARED_FLOAT charge;
    SHARED_USHORT time;
    SHARED_UCHAR pad;
    SHARED_UCHAR row;
    SHARED_UCHAR cru;
    /*SHARED_UCHAR padding[3]; Implicit padding to keep struct 4 byte aligned*/
} PaddedDigit;

#define PADDED_DIGIT_SIZE 12


typedef struct PackedDigit_s
{
    SHARED_FLOAT charge;
    SHARED_USHORT time;
    SHARED_UCHAR pad;
    SHARED_UCHAR row;
} PackedDigit;

#define PACKED_DIGIT_SIZE 8


#endif //!defined(SHARED_DIGIT_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
