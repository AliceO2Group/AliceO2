// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_GPU_CLUSTERFINDERCONFIG_H
#define O2_GPU_CLUSTERFINDERCONFIG_H

#ifdef __OPENCL__
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define SCRATCH_PAD_SEARCH_N 8
#define SCRATCH_PAD_COUNT_N 16
#define SCRATCH_PAD_BUILD_N 8
#define SCRATCH_PAD_NOISE_N 8

#include "shared/Digit.h"
#include "shared/tpc.h"


#if defined(CHARGEMAP_TYPE_HALF)
using Charge = half;
#else
using Charge = float;
#endif

#ifndef __OPENCL__
#define LOOP_UNROLL_ATTR
#elif defined(UNROLL_LOOPS)
#define LOOP_UNROLL_ATTR __attribute__((opencl_unroll_hint))
#else
#define LOOP_UNROLL_ATTR __attribute__((opencl_unroll_hint(1)))
#endif

/*
size_t safeIdx(GlobalPad gpad, Timestamp time)
{
  size_t allElements = TPC_MAX_TIME_PADDED * TPC_NUM_OF_PADS;

  size_t id = get_global_linear_id();

  if (gpad >= TPC_NUM_OF_PADS) {
    printf("%lu: gpad = %hu\n", id, gpad);
    return chargemapIdx(0, 0);
  }

  if (time + PADDING_TIME >= TPC_MAX_TIME_PADDED) {
    printf("%lu: time = %hu\n", id, time);
    return chargemapIdx(0, 0);
  }

  size_t ind = idxTiling8x8(gpad, time);

  if (ind >= allElements) {
    printf("%lu: gpad=%hu, time=%hu, elems = %lu, ind=%lu\n",
           id, gpad, time, allElements, ind);
    return chargemapIdx(0, 0);
  }

  return ind;
}*/


#endif //!defined(CONFIG_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
