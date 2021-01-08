// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDIncludes.h
/// \author David Rohr

#ifndef O2_GPU_GPURECONSTRUCTIONCUDAINCLUDES_H
#define O2_GPU_GPURECONSTRUCTIONCUDAINCLUDES_H

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <cub/cub.cuh>
#include <cub/block/block_scan.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#pragma GCC diagnostic pop
#include <sm_20_atomic_functions.h>

#endif
