#if !defined(SHARED_CONSTANTS_H)
#    define  SHARED_CONSTANTS_H

#include "types.h"

#if IS_CL_DEVICE
# define CONSTANT constant
#else
# define CONSTANT static const
#endif

CONSTANT SHARED_FLOAT CHARGE_THRESHOLD = 2.f;
CONSTANT SHARED_FLOAT OUTER_CHARGE_THRESHOLD = 0.f;
CONSTANT SHARED_FLOAT QTOT_THRESHOLD = 500.f;

#endif //!defined(SHARED_CONSTANTS_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
