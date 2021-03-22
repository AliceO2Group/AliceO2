// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0Calibration/FT0ChannelDataTimeSlotContainer.h"
#include <numeric>
#include <algorithm>

using namespace o2::calibration::fit;

uint64_t FT0ChannelDataTimeSlotContainer::instanceCounter = 0;

FT0ChannelDataTimeSlotContainer::FT0ChannelDataTimeSlotContainer(const FT0CalibrationObject& calibObject,
                                                                 std::size_t minEntries, std::size_t additionalTimeGuard)
  : mCalibrationObject(calibObject), mMinEntries(minEntries), mCreationTimestamp(std::chrono::system_clock::now()),
    mAdditionalTimeGuardInSec(additionalTimeGuard)
{

  mChannelsTimeHistogram = std::make_shared<TH2I>((std::string("Channels_time_histogram") + std::to_string(instanceCounter)).c_str(),
                                                  "Channels_time_histogram",
                                                  NUMBER_OF_HISTOGRAM_BINS, -HISTOGRAM_RANGE, HISTOGRAM_RANGE,
                                                  o2::ft0::Nchannels_FT0, 0, o2::ft0::Nchannels_FT0 - 1);
  mEntriesPerChannel.fill(0);
  ++instanceCounter;
}


bool FT0ChannelDataTimeSlotContainer::hasEnoughEntries() const
{


  auto min_elem = std::min_element(mEntriesPerChannel.begin(), mEntriesPerChannel.end());
  if(*min_elem > mMinEntries){
    return true;
  }

// additional time guard?
  return std::chrono::duration_cast<std::chrono::seconds>
           (std::chrono::system_clock::now() - mCreationTimestamp).count() > mAdditionalTimeGuardInSec;

}
void FT0ChannelDataTimeSlotContainer::fill(const gsl::span<const FT0CalibrationInfoObject>& data)
{

  for(auto& entry : data){

    const auto& chID = entry.getChannelIndex();
    const auto& chTime = entry.getTime();

    //we should add those dummy values as constexpr in dataformatsFT0
    //i dont really know when should it be marked as invalid
    if(o2::ft0::ChannelData::DUMMY_CHANNEL_ID != chID && o2::ft0::ChannelData::DUMMY_CFD_TIME != chTime){
      mChannelsTimeHistogram->Fill(chTime + mCalibrationObject.mChannelOffsets[chID], chID);
      ++mEntriesPerChannel[chID];
    }
    else{
        LOG(FATAL) << "Invalid channel data";
    }
  }
}

void FT0ChannelDataTimeSlotContainer::merge(FT0ChannelDataTimeSlotContainer* prev)
{

  mChannelsTimeHistogram->Add(prev->mChannelsTimeHistogram.get());
  for(unsigned int iCh = 0; iCh < o2::ft0::Nchannels_FT0; ++iCh){
    mEntriesPerChannel[iCh] += prev->mEntriesPerChannel[iCh];
  }
  mCreationTimestamp = prev->mCreationTimestamp;

}

int16_t FT0ChannelDataTimeSlotContainer::getAverageTimeForChannel(std::size_t channelID) const
{

    std::unique_ptr<TH1> projectionX(mChannelsTimeHistogram->ProjectionX("", channelID + 1, channelID + 1));
    return projectionX->GetMean();
}

std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TH1>> FT0ChannelDataTimeSlotContainerViewer::
  generateHistogramForValidChannels(const FT0ChannelDataTimeSlotContainer& obj)
{
  std::map<std::string, std::string> metadata;
  auto histogram = std::shared_ptr<TH1>( obj.getChannelsTimeHistogram()->ProjectionX() );

  auto clName = o2::utils::MemFileHelper::getClassName(*histogram);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);

  return {{TIME_HISTOGRAM_PATH, clName, flName, metadata,
           ccdb::getCurrentTimestamp(), -1}, histogram};
}

std::pair<o2::ccdb::CcdbObjectInfo, std::shared_ptr<TH2>> FT0ChannelDataTimeSlotContainerViewer::
generate2DHistogramTimeInFunctionOfChannel(const FT0ChannelDataTimeSlotContainer& obj)
{

  std::map<std::string, std::string> metadata;
  auto histogram = std::shared_ptr<TH2>( reinterpret_cast<TH2I*>(obj.getChannelsTimeHistogram()->Clone()) );

  auto clName = o2::utils::MemFileHelper::getClassName(*histogram);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);

  return {{TIME_IN_FUNCTION_OF_CHANNEL_HISTOGRAM_PATH, clName, flName, metadata,
           ccdb::getCurrentTimestamp(), -1}, histogram};

}
