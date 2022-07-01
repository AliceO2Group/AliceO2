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
    auto dx = x + rangeX;
    uint32_t binx = dx < 0 ? 0xffffffff : (x + rangeX) * v2BinX;
    auto dy = y + rangeY;
    uint32_t biny = dy < 0 ? 0xffffffff : (y + rangeY) * v2BinY;
    auto dz = z + rangeZ;
    uint32_t binz = dz < 0 ? 0xffffffff : (z + rangeZ) * v2BinZ;
    if (mVerbose) {
      LOG(info) << "dx = " << dx << ", dy = " << dy << ", dz = " << dz;
      LOG(info) << "rangeX = " << rangeX << ", rangeY = " << rangeY << ", rangeZ = " << rangeZ;
      LOG(info) << "v2BinX = " << v2BinX << ", v2BinY = " << v2BinY << ", v2BinZ = " << v2BinZ;
      LOG(info) << "binx = " << binx << ", biny = " << biny << ", binz = " << binz;
    }
    if (binx < nbinsX && biny < nbinsY && binz < nbinsZ) { // do not account for vertices outside the histo ranges
      histoX[binx]++;
      histoY[biny]++;
      histoZ[binz]++;
      entries++;
      histoVtx.push_back({x, y, z});
    }
  }

  for (int i = 0; i < histoX.size(); i++) {
    LOG(debug) << "histoX, bin " << i << ": entries = " << histoX[i];
  }
  for (int i = 0; i < histoY.size(); i++) {
    LOG(debug) << "histoY, bin " << i << ": entries = " << histoY[i];
  }
  for (int i = 0; i < histoZ.size(); i++) {
    LOG(debug) << "histoZ, bin " << i << ": entries = " << histoZ[i];
  }
}

//_____________________________________________
void MeanVertexData::subtract(const MeanVertexData* prev)
{
  // remove entries from prev

  assert(histoX.size() == prev->histoX.size());
  assert(histoY.size() == prev->histoY.size());
  assert(histoZ.size() == prev->histoZ.size());

  for (int i = histoX.size(); i--;) {
    histoX[i] -= prev->histoX[i];
  }
  for (int i = histoY.size(); i--;) {
    histoY[i] -= prev->histoY[i];
  }
  for (int i = histoZ.size(); i--;) {
    histoZ[i] -= prev->histoZ[i];
  }
  histoVtx.erase(histoVtx.begin(), histoVtx.begin() + prev->entries);
  entries -= prev->entries;
}

//_____________________________________________
void MeanVertexData::merge(const MeanVertexData* prev)
{
  // merge data of 2 slots
  assert(histoX.size() == prev->histoX.size());
  assert(histoY.size() == prev->histoY.size());
  assert(histoZ.size() == prev->histoZ.size());

  for (int i = histoX.size(); i--;) {
    histoX[i] += prev->histoX[i];
    histoY[i] += prev->histoY[i];
    histoZ[i] += prev->histoZ[i];
  }
  histoVtx.insert(histoVtx.end(), prev->histoVtx.begin(), prev->histoVtx.end());
  entries += prev->entries;
}

} // end namespace calibration
} // end namespace o2
