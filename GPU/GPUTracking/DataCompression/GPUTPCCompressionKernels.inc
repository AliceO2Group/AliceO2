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

/// \file GPUTPCCompressionKernels.cxx
/// \author David Rohr

#include "GPUO2DataTypes.h"
#include "GPUParam.h"
#include "GPUTPCGeometry.h"
#include "GPUTPCClusterRejection.h"

namespace o2::gpu
{

template <class T>
GPUdii() void GPUTPCCompression_EncodeUnattached(const GPUParam& param, const o2::tpc::ClusterNative& orgCl, T& c, int outidx, const o2::tpc::ClusterNative* orgClPre = nullptr)
{
  if (param.rec.tpc.compressionTypeMask & GPUSettings::CompressionDifferences) {
    unsigned int lastTime = 0, lastPad = 0;
    if (orgClPre) {
      lastPad = orgClPre->padPacked;
      lastTime = orgClPre->getTimePacked();
    }

    c.padDiffU[outidx] = orgCl.padPacked - lastPad;
    c.timeDiffU[outidx] = (orgCl.getTimePacked() - lastTime) & 0xFFFFFF;
  } else {
    c.padDiffU[outidx] = orgCl.padPacked;
    c.timeDiffU[outidx] = orgCl.getTimePacked();
  }
}

} // namespace o2::gpu
