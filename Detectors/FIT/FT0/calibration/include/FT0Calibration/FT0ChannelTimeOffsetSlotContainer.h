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

#ifndef O2_FT0CHANNELTIMEOFFSETSLOTCONTAINER_H
#define O2_FT0CHANNELTIMEOFFSETSLOTCONTAINER_H

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <gsl/span>
#include "DataFormatsFT0/FT0CalibrationInfoObject.h"
#include "DataFormatsFT0/FT0ChannelTimeCalibrationObject.h"
#include "FT0Base/Geometry.h"
#include "Rtypes.h"
#include <TH1F.h>

namespace o2::ft0
{

class FT0ChannelTimeOffsetSlotContainer final
{

  // ranges to be discussed
  static constexpr int HISTOGRAM_RANGE = 500;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS = HISTOGRAM_RANGE;
  static constexpr int NCHANNELS = o2::ft0::Geometry::Nchannels;

 public:
  explicit FT0ChannelTimeOffsetSlotContainer(std::size_t minEntries);
  FT0ChannelTimeOffsetSlotContainer(FT0ChannelTimeOffsetSlotContainer const&);
  FT0ChannelTimeOffsetSlotContainer(FT0ChannelTimeOffsetSlotContainer&&) = default;
  FT0ChannelTimeOffsetSlotContainer& operator=(FT0ChannelTimeOffsetSlotContainer const&);
  FT0ChannelTimeOffsetSlotContainer& operator=(FT0ChannelTimeOffsetSlotContainer&&) = default;
  [[nodiscard]] bool hasEnoughEntries() const;
  void fill(const gsl::span<const FT0CalibrationInfoObject>& data);
  [[nodiscard]] int16_t getMeanGaussianFitValue(std::size_t channelID) const;
  void merge(FT0ChannelTimeOffsetSlotContainer* prev);
  void print() const;
  FT0ChannelTimeCalibrationObject generateCalibrationObject() const;
  void updateFirstCreation(std::uint64_t creation)
  {
    if (creation < mFirstCreation) {
      mFirstCreation = creation;
      for (int iCh = 0; iCh < NCHANNELS; iCh++) {
        std::string histName = std::string{mHistogram[iCh]->GetName()} + "_" + std::to_string(mFirstCreation);
        mHistogram[iCh]->SetName(histName.c_str());
      }
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
  static int sGausFitBins;

 private:
  std::size_t mMinEntries = 1000;
  std::array<uint64_t, NCHANNELS> mEntriesPerChannel{};
  std::array<std::unique_ptr<TH1F>, NCHANNELS> mHistogram;
  std::uint64_t mFirstCreation = std::numeric_limits<std::uint64_t>::max();
  ClassDefNV(FT0ChannelTimeOffsetSlotContainer, 2);
};

} // namespace o2::ft0

#endif // O2_FT0CHANNELTIMEOFFSETSLOTCONTAINER_H
