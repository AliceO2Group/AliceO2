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

#ifndef O2_FT0TIMEOFFSETSLOTCONTAINER_H
#define O2_FT0TIMEOFFSETSLOTCONTAINER_H

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <gsl/span>
#include "CommonDataFormat/FlatHisto2D.h"
#include "DataFormatsFT0/FT0ChannelTimeCalibrationObject.h"

#include "Rtypes.h"
#include <TH1F.h>

namespace o2::ft0
{

class FT0TimeOffsetSlotContainer final
{
  static constexpr int sNCHANNELS = o2::ft0::Geometry::Nchannels;

 public:
  explicit FT0TimeOffsetSlotContainer(std::size_t minEntries);
  FT0TimeOffsetSlotContainer(FT0TimeOffsetSlotContainer const&);
  FT0TimeOffsetSlotContainer(FT0TimeOffsetSlotContainer&&) = default;
  FT0TimeOffsetSlotContainer& operator=(FT0TimeOffsetSlotContainer const&);
  FT0TimeOffsetSlotContainer& operator=(FT0TimeOffsetSlotContainer&&) = default;
  bool hasEnoughEntries() const;
  void fill(const gsl::span<const float>& data);
  int16_t getMeanGaussianFitValue(std::size_t channelID) const;
  void merge(FT0TimeOffsetSlotContainer* prev);
  void print() const;
  FT0ChannelTimeCalibrationObject generateCalibrationObject() const;

 private:
  std::size_t mMinEntries = 1000;
  bool mIsFirstTF{true};
  std::array<uint64_t, sNCHANNELS> mEntriesPerChannel{};
  o2::dataformats::FlatHisto2D<float> mHistogram; // Contains all information about time spectra
  ClassDefNV(FT0TimeOffsetSlotContainer, 1);
};
} // namespace o2::ft0

#endif // O2_FT0TIMEOFFSETSLOTCONTAINER_H
