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

#ifndef O2_FT0TVXPERBCID
#define O2_FT0TVXPERBCID

#include <bitset>
#include <array>
#include <limits>
#include <TH1F.h>

#include "CommonDataFormat/FlatHisto2D.h"
#include "CommonConstants/LHCConstants.h"
#include "DataFormatsFT0/SpectraInfoObject.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/EventsPerBc.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "TH1F.h"
#include "Rtypes.h"

namespace o2::ft0
{
struct EventsPerBcContainer {
  EventsPerBcContainer(int32_t minAmplitudeSideA, int32_t minAmplitudeSideC, int32_t minSumOfAmplitude) : mMinAmplitudeSideA(minAmplitudeSideA), mMinAmplitudeSideC(minAmplitudeSideC), mMinSumOfAmplitude(minSumOfAmplitude) {}

  size_t getEntries() const { return entries; }
  void print() const;
  void fill(const o2::dataformats::TFIDInfo& ti, const gsl::span<const o2::ft0::Digit> data);
  void merge(const EventsPerBcContainer* prev);

  const int32_t mMinAmplitudeSideA;
  const int32_t mMinAmplitudeSideC;
  const int32_t mMinSumOfAmplitude;

  std::array<double, o2::constants::lhc::LHCMaxBunches> mTvx{0.0};
  size_t entries{0};
  long startTimeStamp{0};
  long stopTimeStamp{0};

  ClassDefNV(EventsPerBcContainer, 1);
};

class EventsPerBcCalibrator final : public o2::calibration::TimeSlotCalibration<o2::ft0::EventsPerBcContainer>
{
  using Slot = o2::calibration::TimeSlot<o2::ft0::EventsPerBcContainer>;
  using TFType = o2::calibration::TFType;
  using EventsHistogram = std::array<double, o2::constants::lhc::LHCMaxBunches>;

 public:
  EventsPerBcCalibrator(uint32_t minNumberOfEntries, int32_t minAmplitudeSideA, int32_t minAmplitudeSideC, int32_t minSumOfAmplitude);

  bool hasEnoughData(const Slot& slot) const override;
  void initOutput() override;
  void finalizeSlot(Slot& slot) override;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) override;

  const std::vector<EventsPerBc>& getTvxPerBc() { return mTvxPerBcs; }
  std::vector<std::unique_ptr<o2::ccdb::CcdbObjectInfo>>& getTvxPerBcCcdbInfo() { return mTvxPerBcInfos; }

 private:
  const uint32_t mMinNumberOfEntries;
  const int32_t mMinAmplitudeSideA;
  const int32_t mMinAmplitudeSideC;
  const int32_t mMinSumOfAmplitude;

  std::vector<EventsPerBc> mTvxPerBcs;
  std::vector<std::unique_ptr<o2::ccdb::CcdbObjectInfo>> mTvxPerBcInfos;

  ClassDefOverride(EventsPerBcCalibrator, 1);
};
} // namespace o2::ft0

#endif
