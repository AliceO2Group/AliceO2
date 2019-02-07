#include "config.h"

kernel
void inclusiveScanStart(
        global const uchar *predicate,
        global       int   *outIdx)
{
    int idx = get_global_id(0);

    outIdx[idx] = predicate[idx];
}

kernel
void inclusiveScanStep(
        global const int *input,
        global       int *output)
{
    int idx = get_global_id(0);
    int offset = get_global_offset(0);
    /* printf("idx = %d, offset = %d\n", idx, offset); */

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

    if (predicate[idx])
    {
        digitsOut[newIdx[idx]-1] = digits[idx];
    }
}
