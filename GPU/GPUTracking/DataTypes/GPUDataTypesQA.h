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

/// \file GPUDataTypesIO.h
/// \author David Rohr

#ifndef GPUDATATYPESQA_H
#define GPUDATATYPESQA_H

#include "GPUCommonDef.h"

#include <cstddef>

namespace o2::gpu::gpudatatypes::gpuqa
{
enum gpuQATaskIds : int32_t {
  tasksNone = 0,
  taskTrackingEff = 1,
  taskTrackingRes = 2,
  taskTrackingResPull = 4,
  taskClusterAttach = 8,
  tasksAllMC = 16 - 1,
  taskTrackStatistics = 16,
  taskClusterCounts = 32,
  taskClusterRejection = 64,
  tasksAll = 128 - 1,
  tasksDefault = tasksAll,
  tasksDefaultPostprocess = tasksDefault & ~taskClusterCounts,
  tasksAllNoQC = tasksAll & ~tasksAllMC,
  tasksAutomatic = -1
};
} // namespace o2::gpu::gpudatatypes::gpuqa

#endif
