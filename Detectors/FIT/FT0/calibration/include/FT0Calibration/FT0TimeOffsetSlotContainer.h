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

#include <bitset>
#include <array>

#include "CommonDataFormat/FlatHisto2D.h"
#include "DataFormatsFT0/SpectraInfoObject.h"

#include "TList.h"

#include "Rtypes.h"
namespace o2::ft0
{

class FT0TimeOffsetSlotContainer final
{
  static constexpr int sNCHANNELS = o2::ft0::Geometry::Nchannels;

 public:
  FT0TimeOffsetSlotContainer(std::size_t minEntries); // constructor is needed due to current version of FITCalibration library, should be removed
  FT0TimeOffsetSlotContainer(FT0TimeOffsetSlotContainer const&) = default;
  FT0TimeOffsetSlotContainer(FT0TimeOffsetSlotContainer&&) = default;
  FT0TimeOffsetSlotContainer& operator=(FT0TimeOffsetSlotContainer const&) = default;
  FT0TimeOffsetSlotContainer& operator=(FT0TimeOffsetSlotContainer&&) = default;
  bool hasEnoughEntries() const;
  void fill(const gsl::span<const float>& data);
  SpectraInfoObject getSpectraInfoObject(std::size_t channelID, TList* listHists) const;
  void merge(FT0TimeOffsetSlotContainer* prev);
  void print() const;
  TimeSpectraInfoObject generateCalibrationObject(long tsStartMS, long tsEndMS, const std::string& pathToHists) const;
  typedef float FlatHistoValue_t;
  typedef o2::dataformats::FlatHisto2D<FlatHistoValue_t> FlatHisto2D_t;

 private:
  // Slot number
  uint8_t mCurrentSlot = 0;
  // Status of channels, pending channels = !(good | bad)
  std::bitset<sNCHANNELS> mBitsetBadChIDs;
  std::bitset<sNCHANNELS> mBitsetGoodChIDs;
  // For hist init, for making hist ranges dynamic
  bool mIsFirstTF{true};
  // For slot finalizing
  bool mIsReady{false};
  // Once it is upper than max entry threshold it stops increasing
  std::array<std::size_t, sNCHANNELS> mArrEntries{};

  // Contains all information about time spectra
  FlatHisto2D_t mHistogram;
  ClassDefNV(FT0TimeOffsetSlotContainer, 1);
};
} // namespace o2::ft0

#endif // O2_FT0TIMEOFFSETSLOTCONTAINER_H
