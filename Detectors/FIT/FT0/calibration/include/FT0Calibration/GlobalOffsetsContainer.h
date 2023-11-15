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

#ifndef O2_GLOBALOFFSETSCONTAINER_H
#define O2_GLOBALOFFSETSCONTAINER_H

#include "DataFormatsFT0/GlobalOffsetsInfoObject.h"
#include "DataFormatsFT0/GlobalOffsetsCalibrationObject.h"
#include "Rtypes.h"
#include <TH1F.h>
#include <array>
#include <vector>
#include <gsl/span>

namespace o2::ft0
{
class GlobalOffsetsContainer final
{
  static constexpr int RANGE = 1000;
  static constexpr unsigned int NBINS = 2 * RANGE;

 public:
  explicit GlobalOffsetsContainer(std::size_t minEntries)
  {
    //    mHisto.resize(NBINS, 0.);
  }

  bool hasEnoughEntries() const;
  void fill(const gsl::span<const GlobalOffsetsInfoObject>& data);
  int getMeanGaussianFitValue() const;
  void merge(GlobalOffsetsContainer* prev);
  void print() const;
  GlobalOffsetsCalibrationObject generateCalibrationObject(long, long, const std::string&) const;
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

 private:
  std::uint64_t mFirstCreation = std::numeric_limits<std::uint64_t>::max();
  std::size_t mMinEntries;
  std::array<int, NBINS> mHisto{};
  int mEntries = 0;

  ClassDefNV(GlobalOffsetsContainer, 2);
};

} // namespace o2::ft0

#endif // O2_GLOBALOFFSETCONTAINER_H
