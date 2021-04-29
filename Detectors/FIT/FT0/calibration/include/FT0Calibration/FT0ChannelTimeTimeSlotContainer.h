// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FT0CHANNELTIMETIMESLOTCONTAINER_H
#define O2_FT0CHANNELTIMETIMESLOTCONTAINER_H

#include <array>
#include <vector>
#include <gsl/span>
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "DataFormatsFT0/RawEventData.h"
#include "Rtypes.h"
#include <boost/histogram.hpp>

namespace o2::ft0
{

class FT0ChannelTimeTimeSlotContainer final
{

  //ranges to be discussed
  static constexpr int HISTOGRAM_RANGE = 200;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS = 2 * HISTOGRAM_RANGE;

  using BoostHistogramType = boost::histogram::histogram<std::tuple<boost::histogram::axis::integer<>,
                                                                    boost::histogram::axis::integer<>>,
                                                         boost::histogram::unlimited_storage<std::allocator<char>>>;

 public:
  explicit FT0ChannelTimeTimeSlotContainer(std::size_t minEntries);
  [[nodiscard]] bool hasEnoughEntries() const;
  void fill(const gsl::span<const FT0CalibrationInfoObject>& data);
  [[nodiscard]] int16_t getMeanGaussianFitValue(std::size_t channelID) const;
  void merge(FT0ChannelTimeTimeSlotContainer* prev);
  void print() const;

 private:
  std::size_t mMinEntries;
  std::array<uint64_t, o2::ft0::Nchannels_FT0> mEntriesPerChannel{};
  BoostHistogramType mHistogram;

  ClassDefNV(FT0ChannelTimeTimeSlotContainer, 1);
};

} // namespace o2::ft0

#endif //O2_FT0CHANNELTIMETIMESLOTCONTAINER_H
