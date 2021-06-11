// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHCalibration/PedestalCalibrator.h"
#include "Framework/Logger.h"
#include "MathUtils/fit.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <cassert>
#include <iostream>
#include <sstream>
//#include <TStopwatch.h>

namespace o2
{
namespace mch
{
namespace calibration
{

using Slot = o2::calibration::TimeSlot<o2::mch::calibration::PedestalData>;
using clbUtils = o2::calibration::Utils;

//_____________________________________________
void PedestalData::fill(const gsl::span<const o2::mch::calibration::PedestalDigit> digits)
{
  mPedestalProcessor.process(digits);
}

//_____________________________________________
void PedestalData::merge(const PedestalData* prev)
{
  // merge data of 2 slots
}

//_____________________________________________
void PedestalData::print() const
{
  LOG(INFO) << "Printing MCH pedestals:";
  std::ostringstream os;
}

//===================================================================

//_____________________________________________
void PedestalCalibrator::initOutput()
{
  // Here we initialize the vector of our output objects
  mBadChannelsVector.reset();
  return;
}

//_____________________________________________
bool PedestalCalibrator::hasEnoughData(const Slot& slot) const
{

  // Checking if all channels have enough data to do calibration.
  // Delegating this to PedestalData

  const o2::mch::calibration::PedestalData* c = slot.getContainer();
  LOG(INFO) << "Checking statistics";
  return (true);
}

//_____________________________________________
void PedestalCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  o2::mch::calibration::PedestalData* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // keep track of first TimeFrame
  if (slot.getTFStart() < mTFStart) {
    mTFStart = slot.getTFStart();
  }

  auto pedestals = c->getPedestals();
  for (auto& p : pedestals) {
    auto& pMat = p.second;
    for (size_t dsId = 0; dsId < pMat.size(); dsId++) {
      auto& pRow = pMat[dsId];
      for (size_t ch = 0; ch < pRow.size(); ch++) {
        auto& pRecord = pRow[ch];

        if (pRecord.mEntries == 0) {
          continue;
        }

        mPedestalsVector.emplace_back(DsChannelId::make(p.first, dsId, ch), pRecord.mPedestal, pRecord.getRms());

        bool bad = true;
        if (pRecord.mPedestal < mPedestalThreshold) {
          if (pRecord.getRms() < mNoiseThreshold) {
            bad = false;
          }
        }

        if (bad) {
          LOG(INFO) << "S " << p.first << "  DS " << dsId << "  CH " << ch
                    << "  ENTRIES " << pRecord.mEntries << "  PED " << pRecord.mPedestal << "  RMS " << pRecord.getRms();
          mBadChannelsVector.getChannels().emplace_back(p.first, dsId, ch);
        }
      }
    }
  }
}

//_____________________________________________
Slot& PedestalCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<PedestalData>());
  return slot;
}

//_____________________________________________
void PedestalCalibrator::endOfStream()
{
  // create the CCDB entry
  auto clName = o2::utils::MemFileHelper::getClassName(mBadChannelsVector);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> md;
  mBadChannelsInfo = CcdbObjectInfo("MCH/BadChannelCalib", clName, flName, md, mTFStart, 99999999999999);
}

} // end namespace calibration
} // end namespace mch
} // end namespace o2
