// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/SubTimeFrameVisitors.h"
#include "Common/SubTimeFrameDataModel.h"
#include "Common/DataModelUtils.h"

#include <O2Device/O2Device.h>

#include <stdexcept>

#include <vector>
#include <deque>

namespace o2 {
namespace DataDistribution {

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataSerializer
////////////////////////////////////////////////////////////////////////////////

void InterleavedHdrDataSerializer::visit(O2SubTimeFrameLinkData& pStfLinkData)
{
  // header
  mMessages.emplace_back(std::move(pStfLinkData.mCruLinkHeader.get_message()));

  // iterate all Hbfs
  std::move(std::begin(pStfLinkData.mLinkDataChunks), std::end(pStfLinkData.mLinkDataChunks),
            std::back_inserter(mMessages));
}

void InterleavedHdrDataSerializer::visit(O2SubTimeFrameCruData& pStfCruData)
{
  // header
  mMessages.emplace_back(std::move(pStfCruData.mCruHeader.get_message()));

  // std::map<unsigned, O2SubTimeFrameLinkData>
  for (auto& lLinkIdData : pStfCruData.mLinkData)
    lLinkIdData.second.accept(*this);
}

void InterleavedHdrDataSerializer::visit(O2SubTimeFrameRawData& pStfRawData)
{
  // header
  mMessages.emplace_back(std::move(pStfRawData.mRawDataHeader.get_message()));

  // std::map<unsigned, O2SubTimeFrameCruData>
  for (auto& lIdCru : pStfRawData.mCruData)
    lIdCru.second.accept(*this);
}

void InterleavedHdrDataSerializer::visit(O2SubTimeFrame& pStf)
{
  // header
  mMessages.emplace_back(std::move(pStf.mStfHeader.get_message()));

  pStf.mRawData.accept(*this);
}

void InterleavedHdrDataSerializer::serialize(O2SubTimeFrame& pStf, O2Device& pDevice, const std::string& pChan,
                                             const int pChanId)
{
  pStf.accept(*this);

  for (auto& lMsg : mMessages)
    pDevice.Send(lMsg, pChan, pChanId);
}

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataDeserializer
////////////////////////////////////////////////////////////////////////////////

void InterleavedHdrDataDeserializer::visit(O2SubTimeFrameLinkData& pStfLinkData)
{
  int ret;
  // header
  FairMQMessagePtr lHdrMsg(mDevice.NewMessageFor(mChan, mChanId));
  if ((ret = mDevice.Receive(lHdrMsg, mChan, mChanId)) < 0)
    throw std::runtime_error("LinkDataHeader receive failed (err = " + std::to_string(ret) + ")");

  pStfLinkData.mCruLinkHeader = std::move(lHdrMsg);

  // iterate all HBFrames
  for (size_t i = 0; i < pStfLinkData.mCruLinkHeader->payloadSize; i++) {
    FairMQMessagePtr lHbfMsg(mDevice.NewMessageFor(mChan, mChanId));

    if ((ret = mDevice.Receive(lHbfMsg, mChan, mChanId)) < 0)
      throw std::runtime_error("STFrame receive failed (err = " + std::to_string(ret) + ")");

    pStfLinkData.mLinkDataChunks.emplace_back(std::move(lHbfMsg));
  }
}

void InterleavedHdrDataDeserializer::visit(O2SubTimeFrameCruData& pStfCruData)
{
  int ret;
  // header
  FairMQMessagePtr lHdrMsg(mDevice.NewMessageFor(mChan, mChanId));
  if ((ret = mDevice.Receive(lHdrMsg, mChan, mChanId)) < 0)
    throw std::runtime_error("CruHeader receive failed (err = " + std::to_string(ret) + ")");

  pStfCruData.mCruHeader = std::move(lHdrMsg);

  // receive all CRU Link data
  // std::map<unsigned, O2SubTimeFrameLinkData>
  for (size_t i = 0; i < pStfCruData.mCruHeader->payloadSize; i++) {
    O2SubTimeFrameLinkData lLinkObj;

    lLinkObj.accept(*this);

    assert(pStfCruData.mLinkData.count(lLinkObj.mCruLinkHeader->mCruLinkId) == 0);
    pStfCruData.mLinkData.insert(std::make_pair(lLinkObj.mCruLinkHeader->mCruLinkId, std::move(lLinkObj)));
  }
}

void InterleavedHdrDataDeserializer::visit(O2SubTimeFrameRawData& pStfRawData)
{
  int ret;
  // header
  FairMQMessagePtr lHdrMsg(mDevice.NewMessageFor(mChan, mChanId));
  if ((ret = mDevice.Receive(lHdrMsg, mChan, mChanId)) < 0)
    throw std::runtime_error("RawDataHeader receive failed (err = " + std::to_string(ret) + ")");

  pStfRawData.mRawDataHeader = std::move(lHdrMsg);

  // receive all data chunks
  // std::map<unsigned, O2SubTimeFrameCruData>
  for (size_t c = 0; c < pStfRawData.Header().payloadSize; c++) {
    O2SubTimeFrameCruData lCruData;

    // descend into CruData Object
    lCruData.accept(*this);

    assert(0 == pStfRawData.mCruData.count(lCruData.mCruHeader->mCruId));
    pStfRawData.mCruData.insert(std::make_pair(lCruData.mCruHeader->mCruId, std::move(lCruData)));
  }
}

void InterleavedHdrDataDeserializer::visit(O2SubTimeFrame& pStf)
{
  int ret;
  // header
  FairMQMessagePtr lHdrMsg(mDevice.NewMessageFor(mChan, mChanId));
  if ((ret = mDevice.Receive(lHdrMsg, mChan, mChanId)) < 0)
    throw std::runtime_error("StfHeader receive failed (err = " + std::to_string(ret) + ")");

  pStf.mStfHeader = std::move(lHdrMsg);

  // descend into sub objects
  pStf.mRawData.accept(*this);
}

bool InterleavedHdrDataDeserializer::deserialize(O2SubTimeFrame& pStf)
{
  try
  {
    pStf.accept(*this);
  }
  catch (std::runtime_error& e)
  {
    LOG(ERROR) << "SubTimeFrame deserialization failed. Reason: " << e.what();
    return false; // TODO: what? O2Device.Receive() does not throw...?
  }
  catch (std::exception& e)
  {
    LOG(ERROR) << "SubTimeFrame deserialization failed. Reason: " << e.what();
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// HdrDataSerializer
////////////////////////////////////////////////////////////////////////////////

void HdrDataSerializer::visit(O2SubTimeFrameLinkData& pStfLinkData)
{
  // header
  mHeaderMessages.emplace_back(std::move(pStfLinkData.mCruLinkHeader.get_message()));

  // iterate all Hbfs
  std::move(std::begin(pStfLinkData.mLinkDataChunks), std::end(pStfLinkData.mLinkDataChunks),
            std::back_inserter(mDataMessages));

  pStfLinkData.mLinkDataChunks.clear();
}

void HdrDataSerializer::visit(O2SubTimeFrameCruData& pStfCruData)
{
  // header
  mHeaderMessages.push_back(std::move(pStfCruData.mCruHeader.get_message()));

  // std::map<unsigned, O2SubTimeFrameLinkData>
  for (auto& lLinkIdData : pStfCruData.mLinkData)
    lLinkIdData.second.accept(*this);
}

void HdrDataSerializer::visit(O2SubTimeFrameRawData& pStfRawData)
{
  // header
  mHeaderMessages.push_back(std::move(pStfRawData.mRawDataHeader.get_message()));
  // std::map<unsigned, O2SubTimeFrameCruData>
  for (auto& lIdCru : pStfRawData.mCruData)
    lIdCru.second.accept(*this);
}

void HdrDataSerializer::visit(O2SubTimeFrame& pStf)
{
  mHeaderMessages.push_back(std::move(pStf.mStfHeader.get_message()));

  pStf.mRawData.accept(*this);
}

void HdrDataSerializer::serialize(O2SubTimeFrame& pStf, O2Device& pDevice, const std::string& pChan, const int pChanId)
{
  mHeaderMessages.clear();
  mDataMessages.clear();
  // add bookkeeping headers to mark how many messages are being sent
  mHeaderMessages.push_back(std::move(pDevice.NewMessageFor(pChan, pChanId, sizeof(std::size_t))));
  mDataMessages.push_back(std::move(pDevice.NewMessageFor(pChan, pChanId, sizeof(std::size_t))));

  // get messages
  pStf.accept(*this);

  // update counters
  std::size_t lHdrCnt = mHeaderMessages.size() - 1;
  memcpy(mHeaderMessages.front()->GetData(), &lHdrCnt, sizeof(std::size_t));
  std::size_t lDataCnt = mDataMessages.size() - 1;
  memcpy(mDataMessages.front()->GetData(), &lDataCnt, sizeof(std::size_t));

  // send headers
  for (auto& lMsg : mHeaderMessages)
    pDevice.Send(lMsg, pChan, pChanId);

  // send data
  for (auto& lMsg : mDataMessages)
    pDevice.Send(lMsg, pChan, pChanId);

  mHeaderMessages.clear();
  mDataMessages.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// HdrDataVisitor
////////////////////////////////////////////////////////////////////////////////

void HdrDataDeserializer::visit(O2SubTimeFrameLinkData& pStfLinkData)
{
  pStfLinkData.mCruLinkHeader = std::move(mHeaderMessages.front());
  mHeaderMessages.pop_front();

  // iterate all HBFrames
  std::move(std::begin(mDataMessages), std::begin(mDataMessages) + pStfLinkData.mCruLinkHeader->payloadSize,
            std::back_inserter(pStfLinkData.mLinkDataChunks));
  mDataMessages.erase(std::begin(mDataMessages), std::begin(mDataMessages) + pStfLinkData.mCruLinkHeader->payloadSize);
}

void HdrDataDeserializer::visit(O2SubTimeFrameCruData& pStfCruData)
{
  pStfCruData.mCruHeader = std::move(mHeaderMessages.front());
  mHeaderMessages.pop_front();

  // receive all CRU Link data
  // std::map<unsigned, O2SubTimeFrameLinkData>
  for (size_t i = 0; i < pStfCruData.mCruHeader->payloadSize; i++) {
    O2SubTimeFrameLinkData lLinkObj;

    lLinkObj.accept(*this);

    assert(pStfCruData.mLinkData.count(lLinkObj.mCruLinkHeader->mCruLinkId) == 0);
    pStfCruData.mLinkData.insert(std::make_pair(lLinkObj.mCruLinkHeader->mCruLinkId, std::move(lLinkObj)));
  }
}

void HdrDataDeserializer::visit(O2SubTimeFrameRawData& pStfRawData)
{
  pStfRawData.mRawDataHeader = std::move(mHeaderMessages.front());
  mHeaderMessages.pop_front();

  // receive all data chunks
  // std::map<unsigned, O2SubTimeFrameCruData>
  for (size_t c = 0; c < pStfRawData.mRawDataHeader->payloadSize; c++) {
    O2SubTimeFrameCruData lCruData;

    // descend into CruData Object
    lCruData.accept(*this);

    assert(0 == pStfRawData.mCruData.count(lCruData.mCruHeader->mCruId));
    pStfRawData.mCruData.insert(std::make_pair(lCruData.mCruHeader->mCruId, std::move(lCruData)));
  }
}

void HdrDataDeserializer::visit(O2SubTimeFrame& pStf)
{
  pStf.mStfHeader = std::move(mHeaderMessages.front());
  mHeaderMessages.pop_front();

  pStf.mRawData.accept(*this);
}

bool HdrDataDeserializer::deserialize(O2SubTimeFrame& pStf)
{
  int ret;

  mHeaderMessages.clear();
  mDataMessages.clear();

  try
  {
    // receive all header messages
    std::size_t lHdrCnt = 0;
    {
      FairMQMessagePtr lHdrCntMsg(mDevice.NewMessageFor(mChan, mChanId));
      if ((ret = mDevice.Receive(lHdrCntMsg, mChan, mChanId)) < 0)
        return false;

      memcpy(&lHdrCnt, lHdrCntMsg->GetData(), sizeof(std::size_t));
    }

    for (size_t h = 0; h < lHdrCnt; h++) {
      FairMQMessagePtr lHdrMsg(mDevice.NewMessageFor(mChan, mChanId));
      if ((ret = mDevice.Receive(lHdrMsg, mChan, mChanId)) < 0)
        return false;

      mHeaderMessages.push_back(std::move(lHdrMsg));
    }

    // receive all data messages
    std::size_t lDataCnt = 0;
    {
      FairMQMessagePtr lDataCntMsg(mDevice.NewMessageFor(mChan, mChanId));
      if ((ret = mDevice.Receive(lDataCntMsg, mChan, mChanId)) < 0)
        return false;

      memcpy(&lDataCnt, lDataCntMsg->GetData(), sizeof(std::size_t));
    }

    for (size_t d = 0; d < lDataCnt; d++) {
      FairMQMessagePtr lDataMsg(mDevice.NewMessageFor(mChan, mChanId));
      if ((ret = mDevice.Receive(lDataMsg, mChan, mChanId)) < 0)
        return false;

      mDataMessages.push_back(std::move(lDataMsg));
    }

    // build the SubtimeFrame
    pStf.accept(*this);

    // cleanup
    mHeaderMessages.clear();
    mDataMessages.clear();
  }
  catch (std::runtime_error& e)
  {
    LOG(ERROR) << "SubTimeFrame deserialization failed. Reason: " << e.what();
    return false; // TODO: what? O2Device.Receive() does not throw...?
  }
  catch (std::exception& e)
  {
    LOG(ERROR) << "SubTimeFrame deserialization failed. Reason: " << e.what();
    return false;
  }
  return true;
}
}
} /* o2::DataDistribution */
