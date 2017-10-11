// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/DataModelUtils.h"
#include "Common/ReadoutDataModel.h"

#include <map>
#include <iterator>
#include <algorithm>

namespace o2 {
namespace DataDistribution {

////////////////////////////////////////////////////////////////////////////////
/// Readout - STF Builder channel multiplexer
////////////////////////////////////////////////////////////////////////////////

bool ReadoutStfBuilderObjectInfo::send(O2Device& pDevice, const std::string& pChan, const int pChanId)
{
  FairMQMessagePtr lInfo(pDevice.NewMessageFor(pChan, pChanId, sizeof(ReadoutStfBuilderObjectInfo)));
  memcpy(lInfo->GetData(), this, sizeof(ReadoutStfBuilderObjectInfo));
  if (pDevice.Send(lInfo, pChan, pChanId) < 0)
    return false;
  else
    return true;
}

bool ReadoutStfBuilderObjectInfo::receive(O2Device& pDevice, const std::string& pChan, const int pChanId)
{
  FairMQMessagePtr lRcvdInfo(pDevice.NewMessageFor(pChan, pChanId));
  if (pDevice.Receive(lRcvdInfo, pChan, pChanId) < 0)
    return false;

  assert(lRcvdInfo->GetSize() == sizeof(ReadoutStfBuilderObjectInfo));
  memcpy(this, lRcvdInfo->GetData(), sizeof(ReadoutStfBuilderObjectInfo)); // !!!cannot use lRcvHeader.headerSize

  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// CRU Link
////////////////////////////////////////////////////////////////////////////////

void O2SubTimeFrameLinkData::accept(ISubTimeFrameVisitor& v)
{
  v.visit(*this);
}

bool O2SubTimeFrameLinkData::send(O2Device& pDevice, const std::string& pChan, const int pChanId)
{
  assert(mCruLinkHeader->payloadSize == mLinkDataChunks.size());

  // send the header
  auto lMsg = std::move(mCruLinkHeader.get_message());
  if (pDevice.Send(lMsg, pChan, pChanId) < 0)
    return false;

// send all data chunks
#if defined(SHM_MULTIPART)
  FairMQParts lMpart;
  lMpart.fParts = std::move(mLinkDataChunks);

  if (!pDevice.Send(lMpart, pChan, pChanId))
    return false;

#else
  for (auto& lDataChunk : mLinkDataChunks) {
    if (pDevice.Send(lDataChunk, pChan, pChanId) < 0)
      return false;
  }

#endif

  mLinkDataChunks.clear();
  return true;
}

bool O2SubTimeFrameLinkData::receive(O2Device& pDevice, const std::string& pChan, const int pChanId)
{
  // clear the state
  mLinkDataChunks.clear();

  // receive the header
  {
    FairMQMessagePtr lRcvdHeader(pDevice.NewMessageFor(pChan, pChanId));
    if (pDevice.Receive(lRcvdHeader, pChan, pChanId) < 0)
      return false;

    mCruLinkHeader = std::move(lRcvdHeader);
  }

// receive all data chunks
#if defined(SHM_MULTIPART)
  FairMQParts lMpart;

  if (pDevice.Receive(lMpart, pChan, pChanId) < 0)
    return false;

  assert(lMpart.Size() == mCruLinkHeader->payloadSize);

  // take ownership back
  mLinkDataChunks = std::move(lMpart.fParts);

#else
  for (size_t i = 0; i < mCruLinkHeader->payloadSize; i++) {
    FairMQMessagePtr lDataMsg(pDevice.NewMessageFor(pChan, pChanId));

    if (pDevice.Receive(lDataMsg, pChan, pChanId) < 0)
      return false;

    mLinkDataChunks.emplace_back(std::move(lDataMsg));
  }
#endif

  return true;
}

std::uint64_t O2SubTimeFrameLinkData::getRawDataSize() const
{
  std::uint64_t lDataSize = 0;

  for (const auto& lDataChunk : mLinkDataChunks) {

    lDataSize += lDataChunk->GetSize();
  }

  return lDataSize;
}
}
} /* o2::DataDistribution */
