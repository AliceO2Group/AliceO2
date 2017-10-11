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
/// CRU
////////////////////////////////////////////////////////////////////////////////

void O2SubTimeFrameCruData::accept(class ISubTimeFrameVisitor& v)
{
  v.visit(*this);
}

std::uint64_t O2SubTimeFrameCruData::getRawDataSize() const
{
  std::uint64_t lDataSize = 0;

  for (const auto& lIdLink : mLinkData) {
    const O2SubTimeFrameLinkData& lLinkData = lIdLink.second;
    lDataSize += lLinkData.getRawDataSize();
  }

  return lDataSize;
}

void O2SubTimeFrameCruData::addCruLinkData(int pChannelId, O2SubTimeFrameLinkData&& pLinkData)
{
  const auto lCruId = pLinkData.mCruLinkHeader->mCruId;
  const auto lCruLinkId = pLinkData.mCruLinkHeader->mCruLinkId;

  if (mLinkData.count(lCruLinkId) == 0) {
    O2SubTimeFrameLinkData& lCruLink = mLinkData[lCruLinkId];

    lCruLink.mCruLinkHeader = std::move(pLinkData.mCruLinkHeader);
    lCruLink.mLinkDataChunks = std::move(pLinkData.mLinkDataChunks);
  } else {
    // just add chunks to the link vector
    O2SubTimeFrameLinkData& lCruLink = mLinkData[lCruLinkId];
    std::move(std::begin(pLinkData.mLinkDataChunks), std::end(pLinkData.mLinkDataChunks),
              std::back_inserter(lCruLink.mLinkDataChunks));
    pLinkData.mLinkDataChunks.clear();
  }

  // Update payload counts
  {
    O2SubTimeFrameLinkData& lCruLink = mLinkData[lCruLinkId];
    lCruLink.mCruLinkHeader->payloadSize = lCruLink.mLinkDataChunks.size();
    mCruHeader->payloadSize = mLinkData.size();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrame Raw data
////////////////////////////////////////////////////////////////////////////////

void O2SubTimeFrameRawData::accept(class ISubTimeFrameVisitor& v)
{
  v.visit(*this);
}

void O2SubTimeFrameRawData::addCruLinkData(int pChannelId, O2SubTimeFrameLinkData&& pLinkData)
{
  const auto lCruId = pLinkData.mCruLinkHeader->mCruId;
  const auto lCruLinkId = pLinkData.mCruLinkHeader->mCruLinkId;

  if (!mRawDataHeader)
    mRawDataHeader = make_channel_ptr<O2StfRawDataHeader>(pChannelId);

  if (mCruData.count(lCruId) == 0) {
    O2SubTimeFrameCruData& lCru = mCruData[lCruId];
    lCru.mCruHeader = make_channel_ptr<O2CruHeader>(pChannelId);

    lCru.mCruHeader->headerSize = sizeof(O2CruHeader);
    lCru.mCruHeader->dataDescription = o2::Header::gDataDescriptionCruData;
    lCru.mCruHeader->dataOrigin = o2::Header::gDataOriginTPC;                           // TODO: others
    lCru.mCruHeader->payloadSerializationMethod = o2::Header::gSerializationMethodNone; // Stf serialization?
    lCru.mCruHeader->payloadSize = 0;                                                   // to hold # of CRUs Links
    lCru.mCruHeader->mCruId = lCruId;

    // update payload
    mRawDataHeader->payloadSize += 1;
  }

  mCruData[lCruId].addCruLinkData(pChannelId, std::move(pLinkData));
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

void O2SubTimeFrame::addCruLinkData(int pChannelId, O2SubTimeFrameLinkData&& pLinkData)
{
  mRawData.addCruLinkData(pChannelId, std::move(pLinkData));
}

void O2SubTimeFrame::getShmRegionMessages(std::map<unsigned, std::vector<FairMQMessagePtr>>& pMessages)
{
  // this feels intrusive...
  for (auto& lIdCru : mRawData.mCruData) {

    unsigned lCruId = lIdCru.first; // used to send to the correct freeing channel

    O2SubTimeFrameCruData& lCruData = lIdCru.second;

    for (auto& lIdChan : lCruData.mLinkData) {
      O2SubTimeFrameLinkData& lLinkData = lIdChan.second;

      std::move(std::begin(lLinkData.mLinkDataChunks), std::end(lLinkData.mLinkDataChunks),
                std::back_inserter(pMessages[lCruId]));

      // clean zombie owners
      lLinkData.mLinkDataChunks.clear();
    }
    lCruData.mLinkData.clear();
  }
  mRawData.mCruData.clear();
}

std::uint64_t O2SubTimeFrame::getRawDataSize() const
{
  std::uint64_t lDataSize = 0;

  for (const auto& lIdCru : mRawData.mCruData) {
    const O2SubTimeFrameCruData& lCruData = lIdCru.second;
    lDataSize += lCruData.getRawDataSize();
  }

  return lDataSize;
}
}
} /* o2::DataDistribution */
