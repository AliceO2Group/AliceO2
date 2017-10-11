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
#include <new>
#include <memory>

#include <gsl/gsl_util>

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameReadoutBuilder
////////////////////////////////////////////////////////////////////////////////

SubTimeFrameReadoutBuilder::SubTimeFrameReadoutBuilder(FairMQChannel& pChan)
  : mChan(pChan),
    mStf(nullptr)
{
}

void SubTimeFrameReadoutBuilder::visit(SubTimeFrame& pStf)
{
  /* empty */
}

void SubTimeFrameReadoutBuilder::addHbFrames(const ReadoutSubTimeframeHeader& pHdr, std::vector<FairMQMessagePtr>&& pHbFrames)
{
  if (!mStf) {
    mStf = std::make_unique<SubTimeFrame>(pHdr.timeframeId);
  }

  const EquipmentIdentifier lEqId = EquipmentIdentifier(
    o2::header::gDataDescriptionRawData,
    o2::header::gDataOriginFLP, // FIXME: proper equipment specification
    pHdr.linkId);

  // NOTE: skip the first message (readout header) in pHbFrames
  for (auto i = 1; i < pHbFrames.size(); i++) {
    DataHeader lDataHdr(
      lEqId.mDataDescription,
      lEqId.mDataOrigin,
      lEqId.mSubSpecification,
      pHbFrames[i]->GetSize());
    lDataHdr.payloadSerializationMethod = gSerializationMethodNone;

    auto lHdrMsg = mChan.NewMessage(sizeof(DataHeader));
    if (!lHdrMsg) {
      LOG(ERROR) << "Allocation error: HbFrame::DataHeader: " << sizeof(DataHeader);
      throw std::bad_alloc();
    }
    std::memcpy(lHdrMsg->GetData(), &lDataHdr, sizeof(DataHeader));

    mStf->addStfData(lDataHdr, SubTimeFrame::StfData{ std::move(lHdrMsg), std::move(pHbFrames[i]) });
  }

  pHbFrames.clear();
}

std::unique_ptr<SubTimeFrame> SubTimeFrameReadoutBuilder::getStf()
{
  return std::move(mStf);
}

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataSerializer
////////////////////////////////////////////////////////////////////////////////

static const o2::header::DataHeader gStfDistDataHeader(
  gDataDescSubTimeFrame,
  o2::header::gDataOriginFLP,
  0, // TODO: subspecification? FLP ID? EPN ID?
  sizeof(SubTimeFrame::Header));

void InterleavedHdrDataSerializer::visit(SubTimeFrame& pStf)
{
  // Pack the Stf header
  auto lDataHeaderMsg = mChan.NewMessage(sizeof(DataHeader));
  if (!lDataHeaderMsg) {
    LOG(ERROR) << "Allocation error: Stf DataHeader::size: " << sizeof(DataHeader);
    throw std::bad_alloc();
  }
  std::memcpy(lDataHeaderMsg->GetData(), &gStfDistDataHeader, sizeof(DataHeader));

  auto lDataMsg = mChan.NewMessage(sizeof(SubTimeFrame::Header));
  if (!lDataMsg) {
    LOG(ERROR) << "Allocation error: Stf::Header::size: " << sizeof(SubTimeFrame::Header);
    throw std::bad_alloc();
  }
  std::memcpy(lDataMsg->GetData(), &pStf.header(), sizeof(SubTimeFrame::Header));

  mMessages.emplace_back(std::move(lDataHeaderMsg));
  mMessages.emplace_back(std::move(lDataMsg));

  for (auto& lDataIdentMapIter : pStf.mData) {
    for (auto& lSubSpecMapIter : lDataIdentMapIter.second) {
      for (auto& lStfDataIter : lSubSpecMapIter.second) {
        mMessages.emplace_back(std::move(lStfDataIter.mHeader));
        mMessages.emplace_back(std::move(lStfDataIter.mData));
      }
    }
  }

  pStf.mData.clear();
  pStf.mHeader = SubTimeFrame::Header();
}

void InterleavedHdrDataSerializer::serialize(std::unique_ptr<SubTimeFrame>&& pStf)
{
  // cleanup
  auto lCleanup = gsl::finally([this] {
    // make sure headers and chunk pointers don't linger
    mMessages.clear();
  });

  assert(mMessages.empty());

  pStf->accept(*this);

  mChan.Send(mMessages);
}

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataDeserializer
////////////////////////////////////////////////////////////////////////////////
void InterleavedHdrDataDeserializer::visit(SubTimeFrame& pStf)
{
  assert(mMessages.size() >= 2); // stf meta messages must be present
  assert(mMessages.size() % 2 == 0);

  // header
  DataHeader lStfDataHdr;
  std::memcpy(&lStfDataHdr, mMessages[0]->GetData(), sizeof(DataHeader));
  // verify the stf DataHeader
  if (!(gStfDistDataHeader == lStfDataHdr)) {
    LOG(WARNING) << "Receiving bad SubTimeFrame::Header::DataHeader message";
    throw std::runtime_error("SubTimeFrame::Header::DataHeader");
  }
  // copy the header
  std::memcpy(&pStf.mHeader, mMessages[1]->GetData(), sizeof(SubTimeFrame::Header));

  // iterate over all incoming HBFrame data sources
  for (auto i = 2; i < mMessages.size(); i += 2) {
    pStf.addStfData({ std::move(mMessages[i]), std::move(mMessages[i + 1]) });
  }
}

std::unique_ptr<SubTimeFrame> InterleavedHdrDataDeserializer::deserialize(FairMQChannel& pChan)
{
  int ret;
  // cleanup
  auto lCleanup = gsl::finally([this] {
    // make sure headers and chunk pointers don't linger
    mMessages.clear();
  });
  assert(mMessages.empty());

  if ((ret = pChan.Receive(mMessages)) < 0) {
    LOG(WARNING) << "STF receive failed (err = " + std::to_string(ret) + ")";
    return nullptr;
  }

  return deserialize_impl();
}

std::unique_ptr<SubTimeFrame> InterleavedHdrDataDeserializer::deserialize(FairMQParts& pMsgs)
{
  // cleanup
  auto lCleanup = gsl::finally([this] {
    // make sure headers and chunk pointers don't linger
    mMessages.clear();
  });
  assert(mMessages.empty());

  mMessages = std::move(pMsgs.fParts);
  pMsgs.fParts.clear();

  return deserialize_impl();
}

std::unique_ptr<SubTimeFrame> InterleavedHdrDataDeserializer::deserialize_impl()
{
  // NOTE: StfID will be updated from the stf header
  std::unique_ptr<SubTimeFrame> lStf = std::make_unique<SubTimeFrame>(0);
  try {
    lStf->accept(*this);
  } catch (std::runtime_error& e) {
    LOG(ERROR) << "SubTimeFrame deserialization failed. Reason: " << e.what();
    return nullptr; // TODO: what? O2Device.Receive() does not throw...?
  } catch (std::exception& e) {
    LOG(ERROR) << "SubTimeFrame deserialization failed. Reason: " << e.what();
    return nullptr;
  }

  return std::move(lStf);
}
}
} /* o2::DataDistribution */
