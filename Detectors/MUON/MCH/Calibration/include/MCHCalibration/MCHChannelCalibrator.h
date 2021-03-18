// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MCHChannelCalibrator.h
/// \brief Implementation of the MCH calibrator using pedestal data
///
/// \author Andrea Ferrero, CEA-Saclay

#ifndef MCH_CHANNEL_CALIBRATOR_H_
#define MCH_CHANNEL_CALIBRATOR_H_

#include "DataFormatsMCH/DsChannelGroup.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "CCDB/CcdbObjectInfo.h"
#include "MCHCalibration/PedestalDigit.h"
#include "MCHCalibration/PedestalProcessor.h"

#include <array>

namespace o2
{
namespace mch
{
namespace calibration
{

/// Implementation of the processor that computes the mean and RMS of the channel-by-channel pedestals
/// from the data corresponding to a time slot.
/// In the MCH case the time slot has by default an infinite duration, therefore there is only a single
/// set of pedestal values for each calibration run.
class MCHChannelData
{
  using Slot = o2::calibration::TimeSlot<o2::mch::calibration::MCHChannelData>;

 public:
  MCHChannelData() = default;
  ~MCHChannelData() = default;

  void print() const;

  /// function to update the pedestal values from the data of a single TimeFrame
  void fill(const gsl::span<const o2::mch::calibration::PedestalDigit> data);
  void merge(const MCHChannelData* prev);

  /// function to access the table of computed pedestals for each readout channel
  const PedestalProcessor::PedestalsMap& getPedestals() { return mPedestalProcessor.getPedestals(); }

 private:
  /// helper class that performs the actual computation of the pedestals from the input digits
  PedestalProcessor mPedestalProcessor;

  ClassDefNV(MCHChannelData, 1);
};

/// Implementation of a calibrator object that checks the computed mean and RMS of the pedestals and compares the
/// values with user-supplied thresholds.
/// The channels whose values exceed one of the thresholds are considered bad/noisy and they are stored into a
/// "bad channels" list that is sent to the CDDB populator.
class MCHChannelCalibrator final : public o2::calibration::TimeSlotCalibration<o2::mch::calibration::PedestalDigit, o2::mch::calibration::MCHChannelData>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<o2::mch::calibration::MCHChannelData>;
  using BadChannelsVector = o2::mch::DsChannelGroup;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;

 public:
  struct ChannelPedestal {
    ChannelPedestal() = default;
    ChannelPedestal(o2::mch::DsChannelId chid, double mean, double rms) : mDsChId(chid), mPedMean(mean), mPedRms(rms) {}

    o2::mch::DsChannelId mDsChId;
    double mPedMean{0};
    double mPedRms{0};
  };
  using PedestalsVector = std::vector<ChannelPedestal>;

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

  const PedestalsVector& getPedestalsVector() const { return mPedestalsVector; }

 private:
  float mNoiseThreshold;
  float mPedestalThreshold;

  TFType mTFStart;

  // output
  BadChannelsVector mBadChannelsVector; /// vector containing the unique IDs of the bad/noisy channels
  CcdbObjectInfo mBadChannelsInfo;      /// vector of CCDB Infos , each element is filled with the CCDB description of the accompanying BadChannelsVector object
  PedestalsVector mPedestalsVector;

  ClassDefOverride(MCHChannelCalibrator, 1);
};

} // end namespace calibration
} // end namespace mch
} // end namespace o2

#endif /* MCH_CHANNEL_CALIBRATOR_H_ */
