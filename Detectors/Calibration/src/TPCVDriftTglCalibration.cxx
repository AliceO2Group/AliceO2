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

#include "DetectorsCalibration/TPCVDriftTglCalibration.h"
#include "Framework/Logger.h"
#include "MathUtils/fit.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <Math/SMatrix.h>

namespace o2
{
namespace calibration
{

using Slot = o2::calibration::TimeSlot<TPCVDTglContainer>;
using clbUtils = o2::calibration::Utils;

//_____________________________________________
void TPCVDriftTglCalibration::initOutput()
{
  // Here we initialize the vector of our output objects
  mVDPerSlot.clear();
  mCCDBInfoPerSlot.clear();
  return;
}

//_____________________________________________
void TPCVDriftTglCalibration::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  const auto* cont = slot.getContainer();
  const auto* h = cont->histo.get();
  auto nx = h->getNBinsX(), ny = h->getNBinsY();
  std::array<double, 3> parg;
  ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>> mat;
  double sS = 0, sX = 0, sY = 0, sXX = 0, sXY = 0;
  int npval = 0;
  for (auto ix = nx; ix--;) {
    const auto sliceY = h->getSliceY(ix);
    auto chi2 = o2::math_utils::fitGaus(ny, sliceY.data(), h->getYMin(), h->getYMax(), parg, &mat);
    if (chi2 < -2) { // failed
      continue;
    }
    double err2 = mat(1, 1);
    if (err2 <= 0.f) {
      continue;
    }
    double err2i = 1. / err2, xb = h->getBinXCenter(ix), xbw = xb * err2i, ybw = parg[1] * err2i;
    sS += err2i;
    sX += xbw;
    sXX += xb * xbw;
    sY += ybw;
    sXY += xb * ybw;
    npval++;
  }
  double det = sS * sXX - sX * sX;
  if (!det || npval < 2) {
    LOG(error) << "VDrift fit failed for slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd()
               << " with " << cont->entries << " entries";
  }
  det = 1. / det;
  double offs = (sXX * sY - sX * sXY) * det, slope = (sS * sXY - sX * sY) * det;
  double offsErr = sXX * det, slopErr = sS * det;
  offsErr = offsErr > 0. ? std::sqrt(offsErr) : 0.;
  slopErr = slopErr > 0. ? std::sqrt(slopErr) : 0.;
  float corrFact = 1. / (1. - slope);
  float corrFactErr = corrFact * corrFact * slopErr;
  mVDPerSlot.emplace_back(corrFact, corrFactErr);
  auto clName = o2::utils::MemFileHelper::getClassName(mVDPerSlot.back());
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> md;
  mCCDBInfoPerSlot.emplace_back("TPC/Calib/VDriftTgl", clName, flName, md,
                                slot.getStartTimeMS() - 10 * o2::ccdb::CcdbObjectInfo::SECOND, slot.getEndTimeMS() + o2::ccdb::CcdbObjectInfo::MONTH);
  LOGP(info, "Finalize slot {}({})<=TF<={}({}) with {} entries | dTgl vs Tgl_ITS offset: {}+-{} Slope: {}+-{} -> VD corr factor = {}+-{}", slot.getTFStart(), slot.getStartTimeMS(),
       slot.getTFEnd(), slot.getEndTimeMS(), cont->entries, offs, offsErr, slope, slopErr, corrFact, corrFactErr);
}

//_____________________________________________
Slot& TPCVDriftTglCalibration::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<TPCVDTglContainer>(mNBinsTgl, mMaxTgl, mNBinsDTgl, mMaxDTgl));
  return slot;
}

} // end namespace calibration
} // end namespace o2
