// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef MCH_CHANNEL_CALIBRATOR_H_
#define MCH_CHANNEL_CALIBRATOR_H_

#include "DataFormatsMCH/DsChannelGroup.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "CCDB/CcdbObjectInfo.h"
#include "MCHCalibration/PedestalDigit.h"
#include "MCHCalibration/PedestalProcessor.h"

#include <array>
#include <boost/histogram.hpp>

namespace o2
{
namespace mch
{
namespace calibration
{

class MCHChannelData
{

  using Slot = o2::calibration::TimeSlot<o2::mch::calibration::MCHChannelData>;

 public:
  MCHChannelData() = default;
  ~MCHChannelData() = default;

  void print() const;
  void fill(const gsl::span<const o2::mch::calibration::PedestalDigit> data);
  void merge(const MCHChannelData* prev);

  //const std::vector<o2::mch::calibration::PedestalDigit>& getDigits() const { return mDigits; }
  const PedestalProcessor::PedestalsMap& getPedestals() { return mPedestalProcessor.getPedestals(); }

 private:
  PedestalProcessor mPedestalProcessor;

  ClassDefNV(MCHChannelData, 1);
};

class MCHChannelCalibrator final : public o2::calibration::TimeSlotCalibration<o2::mch::calibration::PedestalDigit, o2::mch::calibration::MCHChannelData>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<o2::mch::calibration::MCHChannelData>;
  using BadChannelsVector = o2::mch::DsChannelGroup;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

 public:
  MCHChannelCalibrator(float pedThreshold, float noiseThreshold) : mPedestalThreshold(pedThreshold), mNoiseThreshold(noiseThreshold), mTFStart(0xffffffffffffffff){};

  ~MCHChannelCalibrator() final = default;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  void endOfStream();

  const BadChannelsVector& getBadChannelsVector() const { return mBadChannelsVector; }
  const CcdbObjectInfo& getBadChannelsInfo() const { return mBadChannelsInfo; }
  CcdbObjectInfo& getBadChannelsInfo() { return mBadChannelsInfo; }

 private:
  float mNoiseThreshold;
  float mPedestalThreshold;

  TFType mTFStart;

  // output
  BadChannelsVector mBadChannelsVector;
  CcdbObjectInfo mBadChannelsInfo; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying TimeSlewing object

  ClassDefOverride(MCHChannelCalibrator, 1);
};

} // end namespace calibration
} // end namespace mch
} // end namespace o2

#endif /* MCH_CHANNEL_CALIBRATOR_H_ */
