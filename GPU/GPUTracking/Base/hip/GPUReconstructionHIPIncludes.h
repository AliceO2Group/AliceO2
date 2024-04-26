// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionHIPInclude.h
/// \author David Rohr

#ifndef O2_GPU_RECONSTRUCTIONHIPINCLUDES_H
#define O2_GPU_RECONSTRUCTIONHIPINCLUDES_H

#define __HIP_ENABLE_DEVICE_MALLOC__ 1 // Fix SWDEV-239120

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hipcub/hipcub.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#pragma GCC diagnostic pop

#endif
