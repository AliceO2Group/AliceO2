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
  bool badMapBool[Geometry::kNCHANNELS] = {};

  // handle data from pedestal run first
  // check pedestal efficiencies
  if (mPedEfficiencies) {
    LOG(info) << "NoiseCalibrator::finalizeSlot() : checking ped efficiencies";
    for (int i = 0; i < Geometry::kNCHANNELS; i++) {
      badMapBool[i] = false;
      if ((*mPedEfficiencies.get())[i] > mToleratedChannelEfficiencyHigh ||
          (*mPedEfficiencies.get())[i] < mToleratedChannelEfficiencyLow) {
        badMapBool[i] = true;
      }
    }
  }

  // check dead channels
  if (mDeadChannels) {
    LOG(info) << "NoiseCalibrator::finalizeSlot() : checking dead channels";
    for (unsigned int i = 0; i < mDeadChannels.get()->size(); i++) {
      badMapBool[(*mDeadChannels.get())[i]] = true;
    }
  }

  // check channels with very high pedestal value (> 511)
  if (mHighPedChannels) {
    LOG(info) << "NoiseCalibrator::finalizeSlot() : checking high ped channels";
    for (unsigned int i = 0; i < mHighPedChannels.get()->size(); i++) {
      badMapBool[(*mHighPedChannels.get())[i]] = true;
    }
  }

  // find noisy channels
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    if (float(calibData->mOccupancyMap[i]) / calibData->mNEvents > mNoiseFrequencyCriteria) {
      badMapBool[i] = true;
    }
  }

  // fill BadChannelMap and send it to output
  for (unsigned short i = 0; i < Geometry::kNCHANNELS; i++) {
    if (badMapBool[i]) {
      badMap->addBadChannel(i);
    }
  }
  mBadChannelMapVec.push_back(*badMap);
  // metadata for o2::cpv::BadChannelMap
  std::map<std::string, std::string> metaData;
  auto className = o2::utils::MemFileHelper::getClassName(badMap);
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
