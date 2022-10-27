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

#include "Framework/Logger.h"
#include "CPVCalibration/NoiseCalibrator.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"
#include "CPVBase/Geometry.h"
#include "CPVBase/CPVCalibParams.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"

namespace o2
{
namespace cpv
{
using NoiseTimeSlot = o2::calibration::TimeSlot<o2::cpv::NoiseCalibData>;
// NoiseCalibData
//_____________________________________________________________________________
NoiseCalibData::NoiseCalibData(float noiseThreshold)
{
  mNoiseThreshold = noiseThreshold;
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    mOccupancyMap.push_back(0);
  }
}
//_____________________________________________________________________________
void NoiseCalibData::fill(const gsl::span<const Digit> digits)
{
  for (auto& dig : digits) {
    if (dig.getAmplitude() > mNoiseThreshold) {
      mOccupancyMap[dig.getAbsId()]++;
    }
  }
  mNEvents++;
}
//_____________________________________________________________________________
void NoiseCalibData::merge(const NoiseCalibData* prev)
{
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    mOccupancyMap[i] += prev->mOccupancyMap[i];
  }
  mNEvents += prev->mNEvents;
  LOG(info) << "Merged TimeSlot with previous one. Now we have " << mNEvents << " events.";
}
//_____________________________________________________________________________
void NoiseCalibData::print()
{
  LOG(info) << "NoiseCalibData::mNEvents = " << mNEvents;
}
//_____________________________________________________________________________
// NoiseCalibrator
//_____________________________________________________________________________
NoiseCalibrator::NoiseCalibrator()
{
  LOG(info) << "NoiseCalibrator::NoiseCalibrator() : "
            << "Noise calibrator created!";
}
//_____________________________________________________________________________
void NoiseCalibrator::configParameters()
{
  auto& cpvParams = CPVCalibParams::Instance();
  mMinEvents = cpvParams.noiseMinEvents;
  mToleratedChannelEfficiencyLow = cpvParams.noiseToleratedChannelEfficiencyLow;
  mToleratedChannelEfficiencyHigh = cpvParams.noiseToleratedChannelEfficiencyHigh;
  mNoiseFrequencyCriteria = cpvParams.noiseFrequencyCriteria;
  mNoiseThreshold = cpvParams.noiseThreshold;
  LOG(info) << "NoiseCalibrator::configParameters() : following parameters configured:";
  LOG(info) << "mMinEvents = " << mMinEvents;
  LOG(info) << "mToleratedChannelEfficiencyLow = " << mToleratedChannelEfficiencyLow;
  LOG(info) << "mToleratedChannelEfficiencyHigh = " << mToleratedChannelEfficiencyHigh;
  LOG(info) << "mNoiseFrequencyCriteria = " << mNoiseFrequencyCriteria;
  LOG(info) << "mNoiseThreshold = " << mNoiseThreshold;
}
//_____________________________________________________________________________
void NoiseCalibrator::initOutput()
{
  LOG(info) << "NoiseCalibrator::initOutput() : output vectors cleared";
  mCcdbInfoBadChannelMapVec.clear();
  mBadChannelMapVec.clear();
}
//_____________________________________________________________________________
void NoiseCalibrator::finalizeSlot(NoiseTimeSlot& slot)
{
  NoiseCalibData* calibData = slot.getContainer();
  LOG(info) << "NoiseCalibrator::finalizeSlot() : finalizing slot "
            << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " with " << calibData->mNEvents << " events.";
  o2::cpv::BadChannelMap* badMap = new o2::cpv::BadChannelMap();
  bool badMapBool[Geometry::kNCHANNELS] = {false};

  // persistent bad channels from ccdb
  if (mPersistentBadChannels) {
    LOG(info) << "NoiseCalibrator::finalizeSlot() : adding " << mPersistentBadChannels->size() << " permanent bad channels";
    for (int i = 0; i < mPersistentBadChannels->size(); i++) {
      badMapBool[(*mPersistentBadChannels)[i]] = true;
    }
  }

  // handle data from pedestal run first
  // check pedestal efficiencies
  int badEfficiencyChannels = 0;
  if (mPedEfficiencies) {
    LOG(info) << "NoiseCalibrator::finalizeSlot() : checking ped efficiencies from pedestal run";
    for (int i = 0; i < Geometry::kNCHANNELS; i++) {
      if ((*mPedEfficiencies)[i] > mToleratedChannelEfficiencyHigh ||
          (*mPedEfficiencies)[i] < mToleratedChannelEfficiencyLow) {
        badMapBool[i] = true;
        badEfficiencyChannels++;
      }
    }
    LOG(info) << "NoiseCalibrator::finalizeSlot() : found " << badEfficiencyChannels << " bad ped efficiency channels";
  }

  // check dead channels
  if (mDeadChannels) {
    LOG(info) << "NoiseCalibrator::finalizeSlot() : adding " << mDeadChannels->size() << " dead channels from pedestal run";
    for (int i = 0; i < mDeadChannels->size(); i++) {
      badMapBool[(*mDeadChannels)[i]] = true;
    }
  }

  // check channels with very high pedestal value (> 511)
  if (mHighPedChannels) {
    LOG(info) << "NoiseCalibrator::finalizeSlot() : adding " << mHighPedChannels->size() << " high ped channels from pedestal run";
    for (int i = 0; i < mHighPedChannels->size(); i++) {
      badMapBool[(*mHighPedChannels)[i]] = true;
    }
  }

  // find noisy channels
  int noisyChannels = 0;
  LOG(info) << "NoiseCalibrator::finalizeSlot() : checking noisy channels";
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    if (float(calibData->mOccupancyMap[i]) / calibData->mNEvents > mNoiseFrequencyCriteria) {
      badMapBool[i] = true;
      noisyChannels++;
    }
  }
  LOG(info) << "NoiseCalibrator::finalizeSlot() : found " << noisyChannels << " noisy channels";

  // fill BadChannelMap and send it to output
  int totalBadChannels = 0;
  for (unsigned short i = 0; i < Geometry::kNCHANNELS; i++) {
    if (badMapBool[i]) {
      badMap->addBadChannel(i);
      totalBadChannels++;
    }
  }
  LOG(info) << "NoiseCalibrator::finalizeSlot() : created bad channel map with " << totalBadChannels << " bad channels in total";

  mBadChannelMapVec.push_back(*badMap);
  // metadata for o2::cpv::BadChannelMap
  std::map<std::string, std::string> metaData;
  auto className = o2::utils::MemFileHelper::getClassName(*badMap);
  auto fileName = o2::ccdb::CcdbApi::generateFileName(className);
  auto timeStamp = o2::ccdb::getCurrentTimestamp();
  mCcdbInfoBadChannelMapVec.emplace_back("CPV/Calib/BadChannelMap", className, fileName, metaData, timeStamp, timeStamp + 31536000000); // one year validity time (in milliseconds!)
}
//_____________________________________________________________________________
NoiseTimeSlot& NoiseCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  LOG(info) << "NoiseCalibrator::emplaceNewSlot() : emplacing new Slot from tstart = " << tstart << " to " << tend;
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<NoiseCalibData>(mNoiseThreshold));
  return slot;
}
//_____________________________________________________________________________
} // end namespace cpv
} // end namespace o2
