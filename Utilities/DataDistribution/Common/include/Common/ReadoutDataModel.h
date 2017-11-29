// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_READOUT_DATAMODEL_H_
#define ALICEO2_READOUT_DATAMODEL_H_

#include "Common/Utilities.h"
#include "Common/DataModelUtils.h"

#include "O2Device/O2Device.h"
#include "Headers/DataHeader.h"

#include <vector>
#include <map>
#include <stdexcept>

namespace o2 {
namespace DataDistribution {

using namespace o2::Base;
using namespace o2::Header;

////////////////////////////////////////////////////////////////////////////////
/// Readout - STF Builder channel multiplexer
////////////////////////////////////////////////////////////////////////////////

enum ReadoutStfBuilderObjectType {
  eStfStart,   // info in the object itself
  eReadoutData // an O2SubTimeFrameLinkData follows
};

struct ReadoutStfBuilderObjectInfo : public DataHeader {
  ReadoutStfBuilderObjectInfo() : DataHeader()
  {
    headerSize = sizeof(ReadoutStfBuilderObjectInfo);
    dataDescription = gDataDescriptionInfo;
    payloadSerializationMethod = gSerializationMethodNone;
  }

  ReadoutStfBuilderObjectInfo(O2Device& pDevice, const std::string& pChan, const int pChanId)
  {
    if (!receive(pDevice, pChan, pChanId))
      throw std::runtime_error("receive error");
  }

  bool send(O2Device& pDevice, const std::string& pChan, const int pChanId);
  bool receive(O2Device& pDevice, const std::string& pChan, const int pChanId);

  // multiplex on this
  ReadoutStfBuilderObjectType mObjectType;

  // might as well piggyback this info to avoid dedicated message
  std::uint64_t mStfId;
};

////////////////////////////////////////////////////////////////////////////////
/// CRU Link
////////////////////////////////////////////////////////////////////////////////

// This data struct is transmitted 'atomically' by the Readout
struct O2CruLinkHeader : public DataHeader {
  unsigned mCruId; // keeps track where to return data chunks
  unsigned mCruLinkId;
  std::uint64_t mStfId; // TODO: move to interface
};

class O2SubTimeFrameLinkData : public IDataModelObject {
public:
  O2SubTimeFrameLinkData() = default;
  ~O2SubTimeFrameLinkData() = default;
  // no copy
  O2SubTimeFrameLinkData(const O2SubTimeFrameLinkData&) = delete;
  O2SubTimeFrameLinkData& operator=(const O2SubTimeFrameLinkData&) = delete;
  // default move
  O2SubTimeFrameLinkData(O2SubTimeFrameLinkData&& a) = default;
  O2SubTimeFrameLinkData& operator=(O2SubTimeFrameLinkData&& a) = default;

  /// 'Receive' constructor
  O2SubTimeFrameLinkData(O2Device& pDevice, const std::string& pChan, const int pChanId)
  {
    if (!receive(pDevice, pChan, pChanId))
      throw std::runtime_error("receive error");
  }

  void accept(ISubTimeFrameVisitor&) override {};

  std::uint64_t getRawDataSize() const;

  bool send(O2Device& pDevice, const std::string& pChan, const int pChanId);
  bool receive(O2Device& pDevice, const std::string& pChan, const int pChanId);

  ChannelPtr<O2CruLinkHeader> mCruLinkHeader;
  std::vector<FairMQMessagePtr> mLinkDataChunks; //  HBFrames
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_READOUT_DATAMODEL_H_ */
