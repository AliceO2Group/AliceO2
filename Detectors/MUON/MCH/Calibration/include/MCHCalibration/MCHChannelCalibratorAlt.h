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

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "CCDB/CcdbObjectInfo.h"
#include "MCHCalibration/PedestalDigit.h"

#include <array>
#include <boost/histogram.hpp>

namespace o2
{
namespace mch
{
namespace calibration
{

static const size_t SOLAR_ID_MAX = 100 * 8;

class MCHChannelData
{

  using Slot = o2::calibration::TimeSlot<o2::mch::calibration::MCHChannelData>;
  using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::integer<>>, boost::histogram::unlimited_storage<std::allocator<char>>>;

 public:
  MCHChannelData()
  {
    size_t hiMax = SOLAR_ID_MAX * 40 * 64;
    for (size_t hi = 0; hi < hiMax; hi++) {
      mHisto[hi] = boost::histogram::make_histogram(boost::histogram::axis::integer<>(0, 200)); // bin is defined as [low, high[
    }
  }

  ~MCHChannelData() = default;

  void print() const;
  void fill(const gsl::span<const o2::mch::calibration::PedestalDigit> data);
  void merge(const MCHChannelData* prev);

 private:
  std::array<boostHisto, (SOLAR_ID_MAX + 1) * 40 * 64> mHisto;

  ClassDefNV(MCHChannelData, 1);
};

class MCHChannelCalibrator final : public o2::calibration::TimeSlotCalibration<o2::mch::calibration::PedestalDigit, o2::mch::calibration::MCHChannelData>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<o2::mch::calibration::MCHChannelData>;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;

 public:
  MCHChannelCalibrator(float pedThreshold, float noiseThreshold) : mPedestalThreshold(pedThreshold), mNoiseThreshold(noiseThreshold)
  {
    for (int s = 0; s <= SOLAR_ID_MAX; s++) {
      for (int i = 0; i < 40; i++) {
        for (int j = 0; j < 64; j++) {
          mEntries[s][i][j] = 0;
          mPedestal[s][i][j] = mNoise[s][i][j] = 0;
        }
      }
    }
  };

  ~MCHChannelCalibrator() final = default;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

 private:
  float mNoiseThreshold;
  float mPedestalThreshold;

  uint64_t mEntries[SOLAR_ID_MAX + 1][40][64];
  double mPedestal[SOLAR_ID_MAX + 1][40][64];
  double mNoise[SOLAR_ID_MAX + 1][40][64];

  // output
  CcdbObjectInfoVector mInfoVector; // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying TimeSlewing object

  ClassDefOverride(MCHChannelCalibrator, 1);
};

} // end namespace calibration
} // end namespace mch
} // end namespace o2

#endif /* MCH_CHANNEL_CALIBRATOR_H_ */
