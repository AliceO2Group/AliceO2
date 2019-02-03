#include "config.h"

kernel
void inclusiveScanStart(
        global const uchar *predicate
        global       int   *outIdx)
{
    int idx = get_global_id(0);

    outIdx[idx] = predicate[idx];
}

kernel
void inclusiveScanStep(
        global       int *output,
        global const int *offset)
{
    int idx = get_global_id(0);

    output[idx] += output[idx - offset];
}

kernel
void compactArr(
        global       Digit *digits,
        global const uchar *predicate, 
        global const int   *newIdx)
{
    int idx = get_global_id(0);

    if (predicate[idx])
    {
        digits[newIdx[idx]-1] = digit[idx];
    }
}
