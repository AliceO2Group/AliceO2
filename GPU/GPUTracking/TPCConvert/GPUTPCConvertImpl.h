// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTPCConvertImpl
{
 public:
  GPUd() static void convert(const GPUConstantMem& cm, int slice, int row, float pad, float time, float& x, float& y, float& z)
  {
    if (cm.param.ContinuousTracking) {
      cm.calibObjects.fastTransform->TransformInTimeFrame(slice, row, pad, time, x, y, z, cm.param.continuousMaxTimeBin);
    } else {
      cm.calibObjects.fastTransform->Transform(slice, row, pad, time, x, y, z);
    }
  }
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
