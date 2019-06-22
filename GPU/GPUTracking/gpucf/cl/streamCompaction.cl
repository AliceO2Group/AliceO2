#include "config.h"

#include "debug.h"

#include "shared/ClusterNative.h"


/* kernel */
/* void harrissScan( */
/*         global const int *input, */
/*         global       int *output) */
/* { */
/*     int idx = get_global_id(0); */
/*     int offset = get_global_offset(0); */

/*     SOFT_ASSERT(idx - offset >= 0); */
/*     DBGPR_2("idx = %d, offset = %d", idx, offset); */

/*     output[idx] = input[idx] + input[idx - offset]; */
/* } */


kernel
void nativeScanUpStart(
        global const uchar *predicate,
        global       int   *sums,
        global       int   *incr)
{
    int idx = get_global_id(0);
    int scanRes = work_group_scan_inclusive_add(predicate[idx]);

    /* sums[idx] = scanRes; */

    int lid = get_local_id(0);
    int lastItem = get_local_size(0) - 1;
    int gid = get_group_id(0);

    /* DBGPR_1("ScanUp: idx = %d", idx); */

    if (lid == lastItem)
    {
        incr[gid] = scanRes;
    }
}

kernel 
void nativeScanUp(
        global int *sums,
        global int *incr)
{
    int idx = get_global_id(0);
    int scanRes = work_group_scan_inclusive_add(sums[idx]);

    /* DBGPR_2("ScanUp: idx = %d, res = %d", idx, scanRes); */

    sums[idx] = scanRes;

    int lid = get_local_id(0);
    int lastItem = get_local_size(0) - 1;
    int gid = get_group_id(0);

    /* DBGPR_1("ScanUp: idx = %d", idx); */

    if (lid == lastItem)
    {
        incr[gid] = scanRes;
    }
}

kernel
void nativeScanTop(global int *incr)
{
    int idx = get_global_id(0);

    /* DBGPR_1("ScanTop: idx = %d", idx); */
    
    incr[idx] = work_group_scan_inclusive_add(incr[idx]);
}

kernel
void nativeScanDown(
        global       int *sums,
        global const int *incr)
{
    int gid = get_group_id(0);
    int idx = get_global_id(0);

    int offset = incr[gid];

    sums[idx] += offset;
}


#define COMPACT_ARR(type) \
    kernel \
    void compact ## type( \
            global const type  *in, \
            global       type  *out, \
            global const uchar *predicate,  \
            global       int   *newIdx, \
            global const int   *incr) \
    { \
        int gid = get_group_id(0) - 1; \
        int idx = get_global_id(0); \
        \
        int lastItem = get_global_size(0) - 1; \
        \
        uchar pred = predicate[idx]; \
        int scanRes = work_group_scan_inclusive_add(pred); \
        \
        if (pred || idx == lastItem) \
        { \
            int compIdx = scanRes + incr[gid]; \
            \
            if (pred) \
            { \
                out[compIdx-1] = in[idx]; \
            } \
            \
            if (idx == lastItem) \
            { \
                newIdx[idx] = compIdx; \
            } \
            \
        } \
        \
    } \
    void anonymousFunction()

COMPACT_ARR(Digit);
COMPACT_ARR(ClusterNative);
