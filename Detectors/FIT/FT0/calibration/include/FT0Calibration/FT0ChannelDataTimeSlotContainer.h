// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FT0CHANNELDATATIMESLOTCONTAINER_H
#define O2_FT0CHANNELDATATIMESLOTCONTAINER_H


#include <array>
#include <vector>
#include <gsl/span>
#include "TH1F.h"
#include "TGraph.h"
#include <chrono>
#include "FT0Calibration/FT0CalibrationInfoObject.h"
#include "FT0Calibration/FT0CalibrationObject.h"
#include "DataFormatsFT0/RawEventData.h"
#include "Rtypes.h"
#include "TH2D.h"
#include <boost/histogram.hpp>


namespace o2::ft0
{

class FT0ChannelDataTimeSlotContainer final
{

  //ranges to be discussed
  static constexpr int HISTOGRAM_RANGE = 200;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS = 2 * HISTOGRAM_RANGE;
  static constexpr bool TEST_MODE = true;
  static constexpr unsigned int TIMER_FOR_TEST_MODE = 3;

  using BoostHistogramType = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular
                                                                    <double, boost::use_default, boost::use_default, boost::use_default>,
                                                                    boost::histogram::axis::integer<>>,
                                                         boost::histogram::unlimited_storage<std::allocator<char>>>;



 public:
  explicit FT0ChannelDataTimeSlotContainer(const FT0CalibrationObject& calibrationObject, std::size_t minEntries);
  [[nodiscard]] bool hasEnoughEntries() const;
  void fill(const gsl::span<const FT0CalibrationInfoObject>& data);
  [[nodiscard]] int16_t getAverageTimeForChannel(std::size_t channelID) const;
  void merge(FT0ChannelDataTimeSlotContainer* prev);
  void print() const;


 private:


  const FT0CalibrationObject& mCalibrationObject;
  std::chrono::time_point<std::chrono::system_clock> mCreationTimestamp;
  std::size_t mMinEntries;

  std::array<uint64_t, o2::ft0::Nchannels_FT0> mEntriesPerChannel{};
  BoostHistogramType mHistogram;


 ClassDefNV(FT0ChannelDataTimeSlotContainer, 1);

};

}


#endif //O2_FT0CHANNELDATATIMESLOTCONTAINER_H
