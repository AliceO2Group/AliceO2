// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file DBScan.cxx
/// \brief
///

#include "ITStracking/DBScan.h"
#include "ITStracking/Definitions.h"
#include "GPUCommonMath.h"

namespace o2
{
namespace its
{

Centroid::Centroid(int* indices, float* position)
{
  for (int i{ 0 }; i < 2; ++i) {
    mIndices[i] = indices[i];
  }
  for (int i{ 0 }; i < 3; ++i) {
    mPosition[i] = position[i];
  }
}

float Centroid::ComputeDistance(const Centroid& c1, const Centroid& c2)
{
  return gpu::GPUCommonMath::Sqrt((c1.mPosition[0] - c2.mPosition[0]) * (c1.mPosition[0] - c2.mPosition[0]) +
                                  (c1.mPosition[1] - c2.mPosition[1]) * (c1.mPosition[1] - c2.mPosition[1]) +
                                  (c1.mPosition[2] - c2.mPosition[2]) * (c1.mPosition[2] - c2.mPosition[2]));
}
} // namespace its
} // namespace o2