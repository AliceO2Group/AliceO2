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

#include <array>
#include <vector>
#include <gsl/span>
#include "DataFormatsFT0/RecoCalibInfoObject.h"
#include "DataFormatsFT0/GlobalOffsetsCalibrationObject.h"
#include "Rtypes.h"
#include <boost/histogram.hpp>
#include <TH1F.h>

namespace o2::ft0
{
class GlobalOffsetsContainer final
{

  //ranges to be discussed
  static constexpr int HISTOGRAM_RANGE = 500;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS = 2 * HISTOGRAM_RANGE;

   using BoostHistogramType = boost::histogram::histogram<std::tuple<boost::histogram::axis::integer<>,
                                                                  boost::histogram::axis::integer<>>,
                                                       boost::histogram::unlimited_storage<std::allocator<char>>>;
 public:

  explicit GlobalOffsetsContainer(std::size_t minEntries);
  bool hasEnoughEntries() const;
  void fill(const gsl::span<const o2::ft0::RecoCalibInfoObject>& data);
  int16_t getMeanGaussianFitValue(std::size_t side) const;
  void merge(GlobalOffsetsContainer* prev);
  void print() const;

 private:
  std::size_t mMinEntries;
  std::array<uint64_t, 3> mEntriesCollTime{};
   BoostHistogramType mHistogram;

  ClassDefNV(GlobalOffsetsContainer, 1);
};

} // namespace o2::ft0

#endif //O2_GLOBALOFFSETCONTAINER_H
