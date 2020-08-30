// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDArtcPre2.h
/// \author David Rohr

// Included during RTC compilation

#define GPUCA_CONSMEM_PTR
#define GPUCA_CONSMEM_CALL

#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/block/block_scan.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <sm_20_atomic_functions.h>
