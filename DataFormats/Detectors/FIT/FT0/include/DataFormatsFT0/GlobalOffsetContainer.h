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

#ifndef O2_GLOBALOFFSETCONTAINER_H
#define O2_GLOBALOFFSETCONTAINER_H

#include <array>
#include <vector>
#include <gsl/span>
#include "DataFormatsFT0/GlobalOffsetsCalibrationObject.h"
#include "DataFormatsFT0/RecPoints.h"
#include "Rtypes.h"
#include <boost/histogram.hpp>

namespace o2::ft0
{

class GlobalOffsetContainer final
{

  //ranges to be discussed
  static constexpr int HISTOGRAM_RANGE = 500;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS = 2 * HISTOGRAM_RANGE;

  using BoostHistogramType = boost::histogram::histogram<std::tuple<boost::histogram::axis::integer<>,
                                                                    boost::histogram::axis::integer<>>,
                                                         boost::histogram::unlimited_storage<std::allocator<char>>>;

 public:
  explicit GlobalOffsetContainer(std::size_t minEntries);
  bool hasEnoughEntries() const;
  void fill(const gsl::span<const GlobalOffsetsCalibrationObject>& data);
  int16_t getMeanGaussianFitValue(std::size_t side) const;
  void merge(GlobalOffsetContainer* prev);
  void print() const;

 private:
  std::size_t mMinEntries;
  std::array<uint64_t, 3> mEntriesCollTime{};
  BoostHistogramType mHistogram;

  ClassDefNV(GlobalOffsetContainer, 1);
};

} // namespace o2::ft0

#endif //O2_GLOBALOFFSETCONTAINER_H
