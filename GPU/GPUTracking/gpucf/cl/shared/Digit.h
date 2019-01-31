#if !defined(SHARED_DIGIT_H)
#    define  SHARED_DIGIT_H

#include "types.h"

typedef struct FloatDigit_s
{
    SHARED_FLOAT charge;
    SHARED_USHORT time;
    SHARED_UCHAR pad;
    SHARED_UCHAR cru;
    SHARED_UCHAR row;
    SHARED_UCHAR padding[3]; // Explicit padding to keep struct 4 byte aligned
} FloatDigit;

#define FLOAT_DIGIT_SIZE 12


typedef struct HalfDigit_s
{
    SHARED_HALF charge; 
    SHARED_USHORT time;
    SHARED_USHORT pad;
    SHARED_UCHAR cru;
    SHARED_UCHAR row;
} HalfDigit;

#define HALF_DIGIT_SIZE 8


#endif //!defined(SHARED_DIGIT_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
