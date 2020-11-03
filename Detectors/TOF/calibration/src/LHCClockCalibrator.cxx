// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFCalibration/LHCClockCalibrator.h"
#include "Framework/Logger.h"
#include "MathUtils/fit.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"

namespace o2
{
namespace tof
{

using Slot = o2::calibration::TimeSlot<o2::tof::LHCClockDataHisto>;
using o2::math_utils::fitGaus;
using LHCphase = o2::dataformats::CalibLHCphaseTOF;
using clbUtils = o2::calibration::Utils;

//_____________________________________________
LHCClockDataHisto::LHCClockDataHisto()
{
  LOG(INFO) << "Default c-tor, not to be used";
}

//_____________________________________________
void LHCClockDataHisto::fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data)
{
  // fill container
  for (int i = data.size(); i--;) {
    auto dt = data[i].getDeltaTimePi();
    dt += range;
    if (dt > 0 && dt < 2 * range) {
      histo[int(dt * v2Bin)]++;
      entries++;
    }
  }
}

//_____________________________________________
void LHCClockDataHisto::merge(const LHCClockDataHisto* prev)
{
  // merge data of 2 slots
  for (int i = histo.size(); i--;) {
    histo[i] += prev->histo[i];
  }
  entries += prev->entries;
}

//_____________________________________________
void LHCClockDataHisto::print() const
{
  LOG(INFO) << entries << " entries";
}

//===================================================================

//_____________________________________________
void LHCClockCalibrator::initOutput()
{
  // Here we initialize the vector of our output objects
  mInfoVector.clear();
  mLHCphaseVector.clear();
  return;
}

//_____________________________________________
void LHCClockCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  o2::tof::LHCClockDataHisto* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " with "
            << c->getEntries() << " entries";
  std::vector<float> fitValues;
  float* array = &c->histo[0];
  double fitres = fitGaus(c->nbins, array, -(c->range), c->range, fitValues);
  if (fitres >= 0) {
    LOG(INFO) << "Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
  } else {
    LOG(ERROR) << "Fit failed with result = " << fitres;
  }

  // TODO: the timestamp is now given with the TF index, but it will have
  // to become an absolute time. This is true both for the lhc phase object itself
  // and the CCDB entry
  std::map<std::string, std::string> md;
  LHCphase l;
  l.addLHCphase(slot.getTFStart(), fitValues[1]);
  auto clName = o2::utils::MemFileHelper::getClassName(l);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfoVector.emplace_back("TOF/LHCphase", clName, flName, md, slot.getTFStart(), 99999999999999);
  mLHCphaseVector.emplace_back(l);

  slot.print();
}

//_____________________________________________
Slot& LHCClockCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<LHCClockDataHisto>(mNBins, mRange));
  return slot;
}

} // end namespace tof
} // end namespace o2
