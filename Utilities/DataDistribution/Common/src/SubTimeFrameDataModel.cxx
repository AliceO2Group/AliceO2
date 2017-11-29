// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/SubTimeFrameDataModel.h"

#include <map>
#include <iterator>
#include <algorithm>

namespace o2 {
namespace DataDistribution {


////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameDataSource
////////////////////////////////////////////////////////////////////////////////
void SubTimeFrameDataSource::accept(ISubTimeFrameVisitor& v)
{
  v.visit(*this);
}

void SubTimeFrameDataSource::addHBFrames(int pChannelId, O2SubTimeFrameLinkData&& pLinkData)
{
  if (mHBFrames.empty()) {
    // take over the header message
    assert(!mStfDataSourceHeader);
    mStfDataSourceHeader = make_channel_ptr<StfDataSourceHeader>(pChannelId);

    // TODO: initialize the header properly (slicing)
    memcpy(mStfDataSourceHeader, pLinkData.mCruLinkHeader, sizeof (DataHeader));

    mHBFrames = std::move(pLinkData.mLinkDataChunks);
  } else {
    assert(pLinkData.mCruLinkHeader);
    assert(pLinkData.mCruLinkHeader->dataOrigin == mStfDataSourceHeader->dataOrigin);
    assert(pLinkData.mCruLinkHeader->dataDescription == mStfDataSourceHeader->dataDescription);
    assert(pLinkData.mCruLinkHeader->subSpecification == mStfDataSourceHeader->subSpecification);

    // just add chunks to the link vector
    std::move(std::begin(pLinkData.mLinkDataChunks), std::end(pLinkData.mLinkDataChunks),
              std::back_inserter(mHBFrames));
    pLinkData.mLinkDataChunks.clear();
  }

  // Update payload count
  mStfDataSourceHeader->payloadSize = mHBFrames.size();
}


////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrame
////////////////////////////////////////////////////////////////////////////////

void O2SubTimeFrame::accept(class ISubTimeFrameVisitor& v)
{
  v.visit(*this);
}

O2SubTimeFrame::O2SubTimeFrame(int pChannelId, uint64_t pStfId)
{
  mStfHeader = make_channel_ptr<O2StfHeader>(pChannelId);

  mStfHeader->headerSize = sizeof(O2StfHeader);
  mStfHeader->dataDescription = o2::Header::gDataDescriptionSubTimeFrame;
  mStfHeader->dataOrigin = o2::Header::gDataOriginFLP;
  mStfHeader->payloadSerializationMethod = o2::Header::gSerializationMethodNone; // Stf serialization?
  mStfHeader->payloadSize = 0;                                                   // to hold # of CRUs in the FLP
  mStfHeader->mStfId = pStfId;
}

void O2SubTimeFrame::addHBFrames(int pChannelId, O2SubTimeFrameLinkData&& pLinkData)
{
  // retrieve or add
  mStfReadoutData[pLinkData.mCruLinkHeader->getDataIdentifier()].addHBFrames(pChannelId, std::move(pLinkData));

  // update the count
  mStfHeader->payloadSize = mStfReadoutData.size();
}


// TODO: remove when callbacks are in
void O2SubTimeFrame::getShmRegionMessages(std::map<unsigned, std::vector<FairMQMessagePtr>>& pMessages)
{
  // this feels intrusive...
  for (auto& lReadoutDataKey : mStfReadoutData) {
    auto &lHBFrameData = lReadoutDataKey.second;
    std::move(std::begin(lHBFrameData.mHBFrames), std::end(lHBFrameData.mHBFrames),
                std::back_inserter(pMessages[0]));

    assert(lHBFrameData.mStfDataSourceHeader);
    lHBFrameData.mStfDataSourceHeader->payloadSize = 0;
    lHBFrameData.mHBFrames.clear();
  }
}

std::uint64_t O2SubTimeFrame::getRawDataSize() const
{
  std::uint64_t lDataSize = 0;

  for (auto& lReadoutDataKey : mStfReadoutData) {
    auto &lHBFrameData = lReadoutDataKey.second;
    for (const auto& lHBFrame : lHBFrameData.mHBFrames)
      lDataSize += lHBFrame->GetSize();
  }

  return lDataSize;
}

}
} /* o2::DataDistribution */
