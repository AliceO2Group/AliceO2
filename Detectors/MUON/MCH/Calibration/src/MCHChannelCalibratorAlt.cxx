// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHCalibration/MCHChannelCalibrator.h"
#include "Framework/Logger.h"
#include "MathUtils/fit.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <TStopwatch.h>

namespace o2
{
namespace mch
{
namespace calibration
{

using Slot = o2::calibration::TimeSlot<o2::mch::calibration::MCHChannelData>;
using clbUtils = o2::calibration::Utils;

//_____________________________________________
void MCHChannelData::fill(const gsl::span<const o2::mch::calibration::PedestalDigit> digits)
{
  bool mDebug = true;
  LOG(INFO) << "[MCHChannelData::fill] digits size " << digits.size();

  for (auto& d : digits) {
    auto solarId = d.getSolarId();
    auto dsId = d.getDsId();
    auto channel = d.getChannel();

    size_t hIndex = solarId * 40 * 64 + dsId * 64 + channel;

    for (size_t i = 0; i < d.nofSamples(); i++) {
      const int s = d.getSample(i);
      float w = 1;

      mHisto[hIndex](s);

      /*mEntries[solarId][dsId][channel] += 1;
      uint64_t N = mEntries[solarId][dsId][channel];

      double p0 = mPedestal[solarId][dsId][channel];
      double p = p0 + (s - p0) / N;
      mPedestal[solarId][dsId][channel] = p;

      double M0 = mNoise[solarId][dsId][channel];
      double M = M0 + (s - p0) * (s - p);
      mNoise[solarId][dsId][channel] = M;*/
    }
    /*if (mDebug) {
      double rms = std::sqrt(mNoise[solarId][dsId][channel] / mEntries[solarId][dsId][channel]);
      std::cout << "solarId " << (int)solarId << "  dsId " << (int)dsId << "  ch " << (int)channel << "  nsamples " << d.nofSamples()
            << "  entries "<< mEntries[solarId][dsId][channel] << "  ped "<< mPedestal[solarId][dsId][channel] << "  noise " << mNoise[solarId][dsId][channel] << "  RMS " << rms << std::endl;
    }*/
  }
}

//_____________________________________________
void MCHChannelData::merge(const MCHChannelData* prev)
{
  // merge data of 2 slots
}

//_____________________________________________
void MCHChannelData::print() const
{
  LOG(INFO) << "Printing MCH pedestals:";
  std::ostringstream os;
}

//===================================================================

//_____________________________________________
void MCHChannelCalibrator::initOutput()
{
  // Here we initialize the vector of our output objects
  mInfoVector.clear();
  return;
}

//_____________________________________________
bool MCHChannelCalibrator::hasEnoughData(const Slot& slot) const
{

  // Checking if all channels have enough data to do calibration.
  // Delegating this to MCHChannelData

  const o2::mch::calibration::MCHChannelData* c = slot.getContainer();
  LOG(INFO) << "Checking statistics";
  return (false);
}

//_____________________________________________
void MCHChannelCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  o2::mch::calibration::MCHChannelData* c = slot.getContainer();
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // for the CCDB entry
  //auto clName = o2::utils::MemFileHelper::getClassName(ts);
  //auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  //mInfoVector.emplace_back("MCH/ChannelCalib", clName, flName, md, slot.getTFStart(), 99999999999999);
  //mTimeSlewingVector.emplace_back(ts);
}

//_____________________________________________
Slot& MCHChannelCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<MCHChannelData>());
  return slot;
}

} // end namespace calibration
} // end namespace mch
} // end namespace o2
