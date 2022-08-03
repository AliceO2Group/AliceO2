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

#ifndef O2_FV0CHANNELTIMEOFFSETSLOTCONTAINER_H
#define O2_FV0CHANNELTIMEOFFSETSLOTCONTAINER_H

#include <array>
#include <vector>
#include <gsl/span>
#include "DataFormatsFV0/FV0CalibrationInfoObject.h"
#include "DataFormatsFV0/FV0ChannelTimeCalibrationObject.h"
#include "FV0Base/Constants.h"
#include "Rtypes.h"
#include <boost/histogram.hpp>
namespace o2::fv0
{

class FV0ChannelTimeOffsetSlotContainer final
{

  // ranges to be discussed
  static constexpr int HISTOGRAM_RANGE = 2000;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS = 2 * HISTOGRAM_RANGE;

  using BoostHistogramType = boost::histogram::histogram<std::tuple<boost::histogram::axis::integer<>,
                                                                    boost::histogram::axis::integer<>>,
                                                         boost::histogram::unlimited_storage<std::allocator<char>>>;

 public:
  explicit FV0ChannelTimeOffsetSlotContainer(std::size_t minEntries);
  [[nodiscard]] bool hasEnoughEntries() const;
  void fill(const gsl::span<const FV0CalibrationInfoObject>& data);
  [[nodiscard]] int16_t getMeanGaussianFitValue(std::size_t channelID) const;
  void merge(FV0ChannelTimeOffsetSlotContainer* prev);
  void print() const;
  static int sGausFitBins;
  void updateFirstCreation(std::uint64_t creation)
  {
    if (creation < mFirstCreation) {
      mFirstCreation = creation;
    }
  }
  void resetFirstCreation()
  {
    mFirstCreation = std::numeric_limits<std::uint64_t>::max();
  }
  std::uint64_t getFirstCreation() const
  {
    return mFirstCreation;
  }
  FV0ChannelTimeCalibrationObject generateCalibrationObject() const;

 private:
  std::size_t mMinEntries;
  std::array<uint64_t, Constants::nFv0Channels> mEntriesPerChannel{};
  BoostHistogramType mHistogram;
  std::uint64_t mFirstCreation = std::numeric_limits<std::uint64_t>::max();

  ClassDefNV(FV0ChannelTimeOffsetSlotContainer, 2);
};

} // namespace o2::fv0

#endif // O2_FV0CHANNELTIMEOFFSETSLOTCONTAINER_H
