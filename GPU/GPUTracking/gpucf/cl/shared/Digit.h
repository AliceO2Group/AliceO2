#if !defined(SHARED_DIGIT_H)
#    define  SHARED_DIGIT_H

#include "types.h"

typedef struct Digit_s
{
    SHARED_FLOAT charge;
    SHARED_INT cru;
    SHARED_INT row;
    SHARED_INT pad;
    SHARED_INT time;   

#if IS_CL_HOST
    Digit_s()
    {
    }

    Digit_s(float _charge, int _cru, int _row, int _pad, int _time)
        : charge(_charge)
        , cru(_cru)
        , row(_row)
        , pad(_pad)
        , time(_time)
    {
    }
#endif

} Digit;

#endif //!defined(SHARED_DIGIT_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
