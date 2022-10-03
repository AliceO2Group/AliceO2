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

/// \file GPUTPCConvertImpl.h
/// \author David Rohr

#ifndef O2_GPU_GPUTPCCONVERTIMPL_H
#define O2_GPU_GPUTPCCONVERTIMPL_H

#include "GPUCommonDef.h"
#include "GPUConstantMem.h"
#include "TPCFastTransform.h"
#include "CorrectionMapsHelper.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTPCConvertImpl
{
 public:
  GPUd() static void convert(const GPUConstantMem& GPUrestrict() cm, int slice, int row, float pad, float time, float& GPUrestrict() x, float& GPUrestrict() y, float& GPUrestrict() z)
  {
    if (cm.param.par.continuousTracking) {
      cm.calibObjects.fastTransformHelper->getCorrMap()->TransformInTimeFrame(slice, row, pad, time, x, y, z, cm.param.par.continuousMaxTimeBin);
    } else {
      cm.calibObjects.fastTransformHelper->Transform(slice, row, pad, time, x, y, z);
    }
  }
  GPUd() static void convert(const TPCFastTransform& GPUrestrict() transform, const GPUParam& GPUrestrict() param, int slice, int row, float pad, float time, float& GPUrestrict() x, float& GPUrestrict() y, float& GPUrestrict() z)
  {
    if (param.par.continuousTracking) {
      transform.TransformInTimeFrame(slice, row, pad, time, x, y, z, param.par.continuousMaxTimeBin);
    } else {
      transform.Transform(slice, row, pad, time, x, y, z);
    }
  }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
