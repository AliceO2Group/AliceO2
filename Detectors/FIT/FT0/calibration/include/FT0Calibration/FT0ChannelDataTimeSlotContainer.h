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

namespace o2::calibration::fit
{

class FT0ChannelDataTimeSlotContainer final
{

  //ranges to be discussed
  static constexpr int HISTOGRAM_RANGE = 200;
  static constexpr unsigned int NUMBER_OF_HISTOGRAM_BINS = 2 * HISTOGRAM_RANGE;

 public:
  explicit FT0ChannelDataTimeSlotContainer(const FT0CalibrationObject& calibrationObject, std::size_t minEntries,
                                           std::size_t additionalTimeGuard);
  [[nodiscard]] bool hasEnoughEntries() const;
  void fill(const gsl::span<const FT0CalibrationInfoObject>& data);
  [[nodiscard]] int16_t getAverageTimeForChannel(std::size_t channelID) const;
  [[nodiscard]] const std::shared_ptr<TH2I>& getChannelsTimeHistogram() const { return mChannelsTimeHistogram; }

  void merge(FT0ChannelDataTimeSlotContainer* prev);

  //we have object in ccdb to visualise calib object, can be empty for now
  void print() const {}


 private:

  //Needed to avoid warnings about the same names of the histograms...
  static uint64_t instanceCounter;

  const FT0CalibrationObject& mCalibrationObject;
  std::chrono::time_point<std::chrono::system_clock> mCreationTimestamp;
  std::size_t mMinEntries;

  std::shared_ptr<TH2I> mChannelsTimeHistogram;
  std::array<uint64_t, o2::ft0::Nchannels_FT0> mEntriesPerChannel{};
  std::size_t mAdditionalTimeGuardInSec;


 ClassDefNV(FT0ChannelDataTimeSlotContainer, 1);

};


class FT0ChannelDataTimeSlotContainerViewer
{

  static constexpr const char* TIME_HISTOGRAM_PATH = "FT0/CalibrationHistograms";
  static constexpr const char* TIME_IN_FUNCTION_OF_CHANNEL_HISTOGRAM_PATH = "FT0/2DCalibrationHistograms";

 public:

  static std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TH1>>
    generateHistogramForValidChannels(const FT0ChannelDataTimeSlotContainer& obj);

  static std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TH2>>
    generate2DHistogramTimeInFunctionOfChannel(const FT0ChannelDataTimeSlotContainer& obj);

};


}


#endif //O2_FT0CHANNELDATATIMESLOTCONTAINER_H
