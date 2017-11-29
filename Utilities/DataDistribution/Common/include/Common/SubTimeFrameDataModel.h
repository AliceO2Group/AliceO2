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

using namespace o2::Base;
using namespace o2::Header;

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameDataSource
////////////////////////////////////////////////////////////////////////////////
struct StfDataSourceHeader : public DataHeader {
  //
};

class SubTimeFrameDataSource : public IDataModelObject {
public:
  SubTimeFrameDataSource() = default;
  ~SubTimeFrameDataSource() = default;
  // no copy
  SubTimeFrameDataSource(const SubTimeFrameDataSource&) = delete;
  SubTimeFrameDataSource& operator=(const SubTimeFrameDataSource&) = delete;
  // default move
  SubTimeFrameDataSource(SubTimeFrameDataSource&& a) = default;
  SubTimeFrameDataSource& operator=(SubTimeFrameDataSource&& a) = default;

  // SubTimeFrameDataSource(uint64_t pStfId, cosnt DataIdentifier &pDataIdent, const DataHeader::SubSpecificationType &pDataSubSpec);

  void accept(ISubTimeFrameVisitor&) override;

  void addHBFrames(int pChannelId, O2SubTimeFrameLinkData&& pLinkData);

  ChannelPtr<StfDataSourceHeader> mStfDataSourceHeader;
  std::vector<FairMQMessagePtr>   mHBFrames;
};


////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrame
////////////////////////////////////////////////////////////////////////////////

struct O2StfHeader : public DataHeader {
  std::uint64_t mStfId;
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

  void addHBFrames(int pChannelId, O2SubTimeFrameLinkData&& pLinkData);

  void getShmRegionMessages(std::map<unsigned, std::vector<FairMQMessagePtr>>& pMessages);
  std::uint64_t getRawDataSize() const;

  const O2StfHeader& Header() const
  {
    return *mStfHeader;
  }

  ChannelPtr<O2StfHeader> mStfHeader;

  // 1. map: DataIdentifier -> Data (e.g. (TPC, CLUSTERS) => (All cluster data) )
  // 2. map: SubSpecification -> DataSubset (e.g. (TPC, CLUSTERS, clFinder1000) -> (data from that one source)
  std::map<DataIdentifier, SubTimeFrameDataSource> mStfReadoutData;
};

}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_DATAMODEL_H_ */
