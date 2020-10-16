// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsCalibration/MeanVertexData.h"
#include "Framework/Logger.h"
#include "MathUtils/MathBase.h"
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
using o2::math_utils::math_base::fitGaus;
using clbUtils = o2::calibration::Utils;

//_____________________________________________
MeanVertexData::MeanVertexData()
{
  LOG(INFO) << "Default c-tor, not to be used";
}

//_____________________________________________
void MeanVertexData::print() const
{
  LOG(INFO) << entries << " entries";
}

//_____________________________________________
void MeanVertexData::fill(const gsl::span<const PVertex> data)
{
  // fill container
  LOG(INFO) << "input size = " << data.size();
  for (int i = data.size(); i--;) {
    //if (useFit) {
      // filling the histogram in binned mode
    LOG(INFO) << "i = " << i << " --> x = " << data[i].getX() << ", y = " << data[i].getY() << ", z = " << data[i].getZ();
    auto x = data[i].getX();
      x += rangeX;
      auto y = data[i].getY();
      y += rangeY;
      auto z = data[i].getZ();
      z += rangeZ;
      if (x > 0 && x < 2 * rangeX && y > 0 && y < 2 * rangeY && z > 0 && z < 2 * rangeZ) {
	histoX[int(x * v2BinX)]++;
	histoY[int(y * v2BinY)]++;
	histoZ[int(z * v2BinZ)]++;
	entries++;
      }
      /*    }
    else {
      histoX[entries] = data[i].getX();
      histoY[entries] = data[i].getY();
      histoZ[entries] = data[i].getZ();
      entries++;
    }
      */
  }

  for (int i = 0; i < histoX.size(); i++){
    LOG(INFO) << "histoX, bin " << i << ": entries = " << histoX[i];
  }
}

//_____________________________________________
void MeanVertexData::subtract(const MeanVertexData* prev)
{
  // remove entries from prev
  assert(histoX.size() == histoY.size());
  assert(histoX.size() == histoZ.size());
  assert(prev->histoX.size() == prev->histoY.size());
  assert(prev->histoX.size() == prev->histoZ.size());
  for (int i = histoX.size(); i--;) {
    histoX[i] -= prev->histoX[i];
    histoY[i] -= prev->histoY[i];
    histoZ[i] -= prev->histoZ[i];
  }
  entries -= prev->entries;
}

//_____________________________________________
void MeanVertexData::merge(const MeanVertexData* prev)
{
  // merge data of 2 slots
  assert(histoX.size() == histoY.size());
  assert(histoX.size() == histoZ.size());
  assert(prev->histoX.size() == prev->histoY.size());
  assert(prev->histoX.size() == prev->histoZ.size());

  //  if (useFit) {
    for (int i = histoX.size(); i--;) {
      histoX[i] += prev->histoX[i];
      histoY[i] += prev->histoY[i];
      histoZ[i] += prev->histoZ[i];
    }
    //}
    /*
  else {
    histoX.reserve(histoX.size() + distance(prev->histoX.begin(), prev->histoX.end()));
    histoX.insert(histoX.end(), prev->histoX.begin(), prev->histoX.end());
    histoY.reserve(histoY.size() + distance(prev->histoY.begin(), prev->histoY.end()));
    histoY.insert(histoY.end(), prev->histoY.begin(), prev->histoY.end());
    histoZ.reserve(histoZ.size() + distance(prev->histoZ.begin(), prev->histoZ.end()));
    histoZ.insert(histoZ.end(), prev->histoZ.begin(), prev->histoZ.end());
  }
    */
  entries += prev->entries;
}



} // end namespace calibration
} // end namespace o2
