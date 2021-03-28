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

using namespace o2::ft0;

FT0ChannelDataTimeSlotContainer::FT0ChannelDataTimeSlotContainer(const FT0CalibrationObject& calibObject,
                                                                 std::size_t minEntries)
  : mCalibrationObject(calibObject), mMinEntries(minEntries), mCreationTimestamp(std::chrono::system_clock::now())
{

  mHistogram = boost::histogram::make_histogram(boost::histogram::axis::regular<>(NUMBER_OF_HISTOGRAM_BINS, -HISTOGRAM_RANGE, HISTOGRAM_RANGE),
                                   boost::histogram::axis::integer<>(0, o2::ft0::Nchannels_FT0));
}




bool FT0ChannelDataTimeSlotContainer::hasEnoughEntries() const
{

  if constexpr (!TEST_MODE)
  {
    //we have test data only for 8 channels

    auto min_elem = std::min_element(mEntriesPerChannel.begin(), mEntriesPerChannel.end());
    if(*min_elem > mMinEntries){
      return true;
    }
    return false;
  }
  else{
    return std::chrono::duration_cast<std::chrono::seconds>
             (std::chrono::system_clock::now() - mCreationTimestamp).count() > TIMER_FOR_TEST_MODE;
  }

}
void FT0ChannelDataTimeSlotContainer::fill(const gsl::span<const FT0CalibrationInfoObject>& data)
{

  for(auto& entry : data){

    const auto chID = entry.getChannelIndex();
    const auto chTime = entry.getTime();

    //i dont really know when should it be marked as invalid
    if(o2::ft0::ChannelData::DUMMY_CHANNEL_ID != chID && o2::ft0::ChannelData::DUMMY_CFD_TIME != chTime){
      mHistogram(chTime + mCalibrationObject.mChannelOffsets[chID], chID);
      ++mEntriesPerChannel[chID];
    }
    else{
        LOG(FATAL) << "Invalid channel data";
    }
  }
}

void FT0ChannelDataTimeSlotContainer::merge(FT0ChannelDataTimeSlotContainer* prev)
{

  mHistogram += prev->mHistogram;
  for(unsigned int iCh = 0; iCh < o2::ft0::Nchannels_FT0; ++iCh){
    mEntriesPerChannel[iCh] += prev->mEntriesPerChannel[iCh];
  }

  //will be deleted
  mCreationTimestamp = prev->mCreationTimestamp;
}

int16_t FT0ChannelDataTimeSlotContainer::getAverageTimeForChannel(std::size_t channelID) const
{
    double avg = 0.;
    for (int iBin = 0; iBin < NUMBER_OF_HISTOGRAM_BINS; ++iBin) {
      const auto& v = mHistogram.at(iBin, channelID);
      avg += v * (iBin - HISTOGRAM_RANGE);
    }
    return avg / mEntriesPerChannel[channelID];
}
void FT0ChannelDataTimeSlotContainer::print() const
{
  //Not a great place for printing calib object, but its temp until view workflow will be ready

  if constexpr (TEST_MODE)
  {
    for(unsigned int i = 0; i < o2::ft0::Nchannels_FT0; ++i){
      LOG(INFO) << "ChID: " << i << " " << mCalibrationObject.mChannelOffsets[i];
    }
  }


}
