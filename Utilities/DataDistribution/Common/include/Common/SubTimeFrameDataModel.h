// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_DATAMODEL_H_
#define ALICEO2_SUBTIMEFRAME_DATAMODEL_H_

#include "Common/Utilities.h"
#include "Common/DataModelUtils.h"
#include "Common/ReadoutDataModel.h"

#include <O2Device/O2Device.h>
#include <Headers/DataHeader.h>

#include <map>
#include <stdexcept>

namespace o2 {
namespace DataDistribution {

using O2DataHeader = o2::Header::DataHeader;
using namespace o2::Base;
using namespace o2::Header;

////////////////////////////////////////////////////////////////////////////////
/// CRU
////////////////////////////////////////////////////////////////////////////////

struct O2CruHeader : public O2DataHeader {
  unsigned mCruId; // keeps track where to return data chunks
};

class O2SubTimeFrameCruData : public IDataModelObject {
public:
  O2SubTimeFrameCruData() = default;
  ~O2SubTimeFrameCruData() = default;
  // no copy
  O2SubTimeFrameCruData(const O2SubTimeFrameCruData&) = delete;
  O2SubTimeFrameCruData& operator=(const O2SubTimeFrameCruData&) = delete;
  // default move
  O2SubTimeFrameCruData(O2SubTimeFrameCruData&& a) = default;
  O2SubTimeFrameCruData& operator=(O2SubTimeFrameCruData&& a) = default;

  void accept(ISubTimeFrameVisitor&) override;

  void addCruLinkData(int pChannelId, O2SubTimeFrameLinkData&& pLinkData);
  std::uint64_t getRawDataSize() const;

  ChannelPtr<O2CruHeader> mCruHeader;

  /// map <CRU LINK ID, CRU Link Data>
  std::map<unsigned, O2SubTimeFrameLinkData> mLinkData;
};

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrame Raw data
////////////////////////////////////////////////////////////////////////////////

struct O2StfRawDataHeader : public O2DataHeader {
};

class O2SubTimeFrameRawData : public IDataModelObject {
public:
  O2SubTimeFrameRawData() = default;
  explicit O2SubTimeFrameRawData(int pChannelId) : mRawDataHeader{ make_channel_ptr<O2StfRawDataHeader>(pChannelId) }
  {
  }
  ~O2SubTimeFrameRawData() = default;
  // no copy
  O2SubTimeFrameRawData(const O2SubTimeFrameRawData&) = delete;
  O2SubTimeFrameRawData& operator=(const O2SubTimeFrameRawData&) = delete;
  // default move
  O2SubTimeFrameRawData(O2SubTimeFrameRawData&& a) = default;
  O2SubTimeFrameRawData& operator=(O2SubTimeFrameRawData&& a) = default;

  void accept(ISubTimeFrameVisitor&) override;

  void addCruLinkData(int pChannelId, O2SubTimeFrameLinkData&& pLinkData);

  const O2StfRawDataHeader Header() const
  {
    return *mRawDataHeader;
  }

  ChannelPtr<O2StfRawDataHeader> mRawDataHeader;
  /// map <CRU ID, CRU Data>
  std::map<unsigned, O2SubTimeFrameCruData> mCruData;
};

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrame
////////////////////////////////////////////////////////////////////////////////

struct O2StfHeader : public O2DataHeader {
  std::uint64_t mStfId; // keeps track where to return data chunks
};

class O2SubTimeFrame : public IDataModelObject {
public:
  O2SubTimeFrame() = default;
  ~O2SubTimeFrame() = default;
  // no copy
  O2SubTimeFrame(const O2SubTimeFrame&) = delete;
  O2SubTimeFrame& operator=(const O2SubTimeFrame&) = delete;
  // default move
  O2SubTimeFrame(O2SubTimeFrame&& a) = default;
  O2SubTimeFrame& operator=(O2SubTimeFrame&& a) = default;

  O2SubTimeFrame(int pChannelId, uint64_t pStfId);

  void accept(ISubTimeFrameVisitor&) override;

  void addCruLinkData(int pChannelId, O2SubTimeFrameLinkData&& pLinkData);

  void getShmRegionMessages(std::map<unsigned, std::vector<FairMQMessagePtr>>& pMessages);
  std::uint64_t getRawDataSize() const;

  const O2StfHeader& Header() const
  {
    return *mStfHeader;
  }

  ChannelPtr<O2StfHeader> mStfHeader;
  O2SubTimeFrameRawData mRawData;
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_DATAMODEL_H_ */
