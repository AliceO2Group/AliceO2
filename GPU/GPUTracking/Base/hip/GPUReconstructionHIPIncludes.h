// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionHIPInclude.h
/// \author David Rohr

#ifndef O2_GPU_RECONSTRUCTIONHIPINCLUDES_H
#define O2_GPU_RECONSTRUCTIONHIPINCLUDES_H

#include <hip/hip_runtime.h>
#ifdef __CUDACC__
#define hipExtLaunchKernelGGL(...)
#else
#include <hip/hip_ext.h>
#endif
#include <hipcub/hipcub.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#pragma GCC diagnostic pop

#endif
