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

#include "DetectorsCalibration/MeanVertexData.h"
#include "Framework/Logger.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <cmath>

using namespace o2::calibration;

namespace o2
{
namespace calibration
{

using Slot = o2::calibration::TimeSlot<o2::calibration::MeanVertexData>;
using PVertex = o2::dataformats::PrimaryVertex;
using clbUtils = o2::calibration::Utils;

//_____________________________________________
void MeanVertexData::print() const
{
  LOG(info) << entries << " entries";
}

//_____________________________________________
void MeanVertexData::fill(const gsl::span<const PVertex> data)
{
  // fill container

  LOG(info) << "input size = " << data.size();
  for (int i = data.size(); i--;) {
    // filling the histogram in binned mode
    std::array<float, 3> xyz{data[i].getX(), data[i].getY(), data[i].getZ()};
    auto entries1 = entries + 1;
    for (int j = 0; j < 3; j++) {
      means[j] = (means[j] * entries + xyz[j]) / entries1;
      meanSquares[j] = (meanSquares[j] * entries + xyz[j] * xyz[j]) / entries1;
    }
    if (mVerbose) {
      LOG(info) << "i = " << i << " --> x = " << xyz[0] << ", y = " << xyz[1] << ", z = " << xyz[2];
    }
    entries = entries + 1;
    histoVtx.push_back(xyz);
  }
}

//_____________________________________________
void MeanVertexData::subtract(const MeanVertexData* prev)
{
  // remove entries from prev
  int totEntries = entries - prev->entries;
  if (totEntries > 0) {
    for (int i = 0; i < 3; i++) {
      means[i] = (means[i] * entries - prev->means[i] * prev->entries) / totEntries;
      meanSquares[i] = (meanSquares[i] * entries - prev->meanSquares[i] * prev->entries) / totEntries;
    }
  } else {
    for (int i = 0; i < 3; i++) {
      means[i] = meanSquares[i] = 0.;
    }
  }
  histoVtx.erase(histoVtx.begin(), histoVtx.begin() + prev->entries);
  entries -= prev->entries;
}

//_____________________________________________
void MeanVertexData::merge(const MeanVertexData* prev)
{
  // merge data of 2 slots
  histoVtx.insert(histoVtx.end(), prev->histoVtx.begin(), prev->histoVtx.end());
  auto totEntries = entries + prev->entries;
  if (totEntries) {
    for (int i = 0; i < 3; i++) {
      means[i] = (means[i] * entries + prev->means[i] * prev->entries) / totEntries;
      meanSquares[i] = (meanSquares[i] * entries + prev->meanSquares[i] * prev->entries) / totEntries;
    }
  }
  entries = totEntries;
}

double MeanVertexData::getRMS(int i) const
{
  double rms2 = meanSquares[i] - means[i] * means[i];
  return rms2 > 0. ? std::sqrt(rms2) : 0;
}

} // end namespace calibration
} // end namespace o2
