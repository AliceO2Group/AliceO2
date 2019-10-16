// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef GPUCA_ALIGPUCODE
namespace gpucf
{
#define GPUCF() gpucf::
#define MYMIN() CAMath::Min
#define MYATOMICADD() CAMath::AtomicAdd
#define MYSMEM() GPUsharedref() GPUTPCClusterFinderKernels::GPUTPCSharedMemory
#define MYSMEMR() MYSMEM()&
#else
#define __OPENCL__
#define GPUCF()
#define MYMIN() min
#define MYATOMICADD() atomic_add
#define MYSMEM() int
#define MYSMEMR() MYSMEM()
#endif

#include "config.h"

#include "debug.h"

#include "shared/ClusterNative.h"

/* GPUg() */
/* void harrissScan( */
/*         GPUglobalref() const int *input, */
/*         GPUglobalref()       int *output) */
/* { */
/*     int idx = get_global_id(0); */
/*     int offset = get_global_offset(0); */

/*     SOFT_ASSERT(idx - offset >= 0); */
/*     DBGPR_2("idx = %d, offset = %d", idx, offset); */

/*     output[idx] = input[idx] + input[idx - offset]; */
/* } */

GPUd()
void nativeScanUpStart(int nBlocks, int nThreads, int iBlock, int iThread, MYSMEMR() smem,
        GPUglobalref() const uchar *predicate,
        GPUglobalref()       int   *sums,
        GPUglobalref()       int   *incr)
{
    int idx = get_global_id(0);
    int scanRes = work_group_scan_inclusive_add((int) predicate[idx]); // TODO: Why don't we store scanRes and read it back in compactDigit?
    
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

GPUd()
void nativeScanUp(int nBlocks, int nThreads, int iBlock, int iThread, MYSMEMR() smem,
        GPUglobalref() int *sums,
        GPUglobalref() int *incr)
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

GPUd()
void nativeScanTop(int nBlocks, int nThreads, int iBlock, int iThread, MYSMEMR() smem,GPUglobalref() int *incr)
{
    int idx = get_global_id(0);

    /* DBGPR_1("ScanTop: idx = %d", idx); */
    
    int scanRes = work_group_scan_inclusive_add(incr[idx]);
    incr[idx] = scanRes;
}

GPUd()
void nativeScanDown(int nBlocks, int nThreads, int iBlock, int iThread, MYSMEMR() smem,
        GPUglobalref()       int *sums,
        GPUglobalref() const int *incr,
        unsigned int offset)
{
    int gid = get_group_id(0);
    int idx = get_global_id(0) + offset;

    int shift = incr[gid];

    sums[idx] += shift;
}


GPUd()
void compactDigit(int nBlocks, int nThreads, int iBlock, int iThread, MYSMEMR() smem,
        GPUglobalref() const Digit  *in,
        GPUglobalref()       Digit  *out,
        GPUglobalref() const uchar *predicate,
        GPUglobalref()       int   *newIdx,
        GPUglobalref() const int   *incr)
{
    int gid = get_group_id(0);
    int idx = get_global_id(0);

    int lastItem = get_global_size(0) - 1;

    int pred = predicate[idx];
    int scanRes = work_group_scan_inclusive_add(pred);

    int compIdx = scanRes;
    if (gid) {
      compIdx += incr[gid - 1];
    }

    if (pred)
    {
        out[compIdx-1] = in[idx];
    }

    if (idx == lastItem)
    {
        newIdx[idx] = compIdx; // TODO: Eventually, we can just return the last value, no need to store to memory
    }
}

#ifdef GPUCA_ALIGPUCODE
} // namespace gpucf
#endif

#if !defined(GPUCA_ALIGPUCODE)

GPUg()
void nativeScanUpStart_kernel(
        GPUglobal() const GPUCF()uchar *predicate,
        GPUglobal()       int   *sums,
        GPUglobal()       int   *incr)
{
    GPUshared() MYSMEM() smem;
    GPUCF()nativeScanUpStart(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, predicate, sums, incr);
}

GPUg()
void nativeScanUp_kernel(
        GPUglobal() int *sums,
        GPUglobal() int *incr)
{
    GPUshared() MYSMEM() smem;
    GPUCF()nativeScanUp(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, sums, incr);
}

GPUg()
void nativeScanTop_kernel(GPUglobal() int *incr)
{
    GPUshared() MYSMEM() smem;
    GPUCF()nativeScanTop(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, incr);
}

GPUg()
void nativeScanDown_kernel(
        GPUglobal()       int *sums,
        GPUglobal() const int *incr,
        unsigned int offset)
{
    GPUshared() MYSMEM() smem;
    GPUCF()nativeScanDown(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, sums, incr, offset);
}

GPUg()
void compactDigit_kernel(
        GPUglobal() const GPUCF()Digit  *in,
        GPUglobal()       GPUCF()Digit  *out,
        GPUglobal() const GPUCF()uchar *predicate,
        GPUglobal()       int   *newIdx,
        GPUglobal() const int   *incr)
{
    GPUshared() MYSMEM() smem;
    GPUCF()compactDigit(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, in, out, predicate, newIdx, incr);
}

#endif

#undef GPUCF
#undef MYMIN
#undef MYATOMICADD
#undef MYSMEM
#undef MYSMEMR
