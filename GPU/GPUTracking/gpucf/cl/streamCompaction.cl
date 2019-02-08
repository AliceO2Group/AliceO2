#include "config.h"

#include "debug.h"


kernel
void inclusiveScanStep(
        global const int *input,
        global       int *output)
{
    int idx = get_global_id(0);
    int offset = get_global_offset(0);

    SOFT_ASSERT(idx - offset >= 0);
    DBGPR_2("idx = %d, offset = %d", idx, offset);

    output[idx] = input[idx] + input[idx - offset];
}

kernel
void compactArr(
        global const Digit *digits,
        global       Digit *digitsOut,
        global const int   *predicate, 
        global const int   *newIdx)
{
    int idx = get_global_id(0);

    SOFT_ASSERT(newIdx[idx]-1 <= idx);

    if (predicate[idx])
    {
        digitsOut[newIdx[idx]-1] = digits[idx];
    }
}
