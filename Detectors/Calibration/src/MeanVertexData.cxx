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

using namespace o2::calibration;

namespace o2
{
namespace calibration
{

using Slot = o2::calibration::TimeSlot<o2::calibration::MeanVertexData>;
using PVertex = o2::dataformats::PrimaryVertex;
using clbUtils = o2::calibration::Utils;

//_____________________________________________
MeanVertexData::MeanVertexData()
{
  LOG(info) << "Default c-tor, not to be used";
}

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
    auto x = data[i].getX();
    auto y = data[i].getY();
    auto z = data[i].getZ();
    if (mVerbose) {
      LOG(info) << "i = " << i << " --> x = " << x << ", y = " << y << ", z = " << z;
    }
    entries++;
    histoVtx.push_back({x, y, z});
  }
}

//_____________________________________________
void MeanVertexData::subtract(const MeanVertexData* prev)
{
  // remove entries from prev

  histoVtx.erase(histoVtx.begin(), histoVtx.begin() + prev->entries);
  entries -= prev->entries;
}

//_____________________________________________
void MeanVertexData::merge(const MeanVertexData* prev)
{
  // merge data of 2 slots
  histoVtx.insert(histoVtx.end(), prev->histoVtx.begin(), prev->histoVtx.end());
  entries += prev->entries;
}

} // end namespace calibration
} // end namespace o2
