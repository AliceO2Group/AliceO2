#include "config.h"

#include "debug.h"


kernel
void harrissScan(
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
void nativeScanUp(
        global int *sums,
        global int *incr)
{
    int idx = get_global_id(0);
    int scanRes = work_group_inclusive_scan_add(sums[idx]);

    sums[idx] = scanRes;

    int lid = get_local_id(0);
    int lastItem = get_local_size(0) - 1;
    int gid = get_group_id(0);

    if (lid == lastItem)
    {
        incr[gid] = scanRes;
    }
}

kernel
void nativeScanTop(global int *incr)
{
    int idx = get_global_id(0);
    
    incr[idx] = work_group_inclusive_scan_add(incr[idx]);
}

kernel
void nativeScanDown(
        global       int *sums,
        global const int *incr)
{
    int gid = get_group_id(0);
    int idx = get_global_id(0);

    int offset = incr[gid];

    input[idx] += offset;
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
