// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_VISITORS_H_
#define ALICEO2_SUBTIMEFRAME_VISITORS_H_

#include "Common/SubTimeFrameDataModel.h"

#include <Headers/DataHeader.h>

class O2Device;

#include <deque>
#include <vector>

namespace o2 {
namespace DataDistribution {

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataSerializer
////////////////////////////////////////////////////////////////////////////////

class InterleavedHdrDataSerializer : public ISubTimeFrameVisitor {
public:
  InterleavedHdrDataSerializer() = default;
  void serialize(O2SubTimeFrame& pStf, O2Device& pDevice, const std::string& pChan, const int pChanId);

  void visit(O2SubTimeFrameLinkData& pStfLinkData) override;
  void visit(O2SubTimeFrameCruData& pStfCruData) override;
  void visit(O2SubTimeFrameRawData& pStfRawData) override;
  void visit(O2SubTimeFrame& pStf) override;

  std::vector<FairMQMessagePtr> mMessages;
};

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataDeserializer
////////////////////////////////////////////////////////////////////////////////

class InterleavedHdrDataDeserializer : public ISubTimeFrameVisitor {
public:
  InterleavedHdrDataDeserializer(O2Device& pDevice, const std::string& pChan, const int pChanId)
    : mDevice{ pDevice }, mChan{ pChan }, mChanId{ pChanId }
  {
  }

  void visit(O2SubTimeFrameLinkData& pStfLinkData) override;
  void visit(O2SubTimeFrameCruData& pStfCruData) override;
  void visit(O2SubTimeFrameRawData& pStfRawData) override;
  void visit(O2SubTimeFrame& pStf) override;

  bool deserialize(O2SubTimeFrame&);

  std::vector<FairMQMessagePtr> mMessages;

private:
  O2Device& mDevice;
  const std::string& mChan;
  const int mChanId;
};

////////////////////////////////////////////////////////////////////////////////
/// HdrDataSerializer
////////////////////////////////////////////////////////////////////////////////

class HdrDataSerializer : public ISubTimeFrameVisitor {
public:
  HdrDataSerializer() = default;
  void serialize(O2SubTimeFrame& pStf, O2Device& pDevice, const std::string& pChan, const int pChanId);

  void visit(O2SubTimeFrameLinkData& pStfLinkData) override;
  void visit(O2SubTimeFrameCruData& pStfCruData) override;
  void visit(O2SubTimeFrameRawData& pStfRawData) override;
  void visit(O2SubTimeFrame& pStf) override;

  std::vector<FairMQMessagePtr> mHeaderMessages;
  std::vector<FairMQMessagePtr> mDataMessages;
};

////////////////////////////////////////////////////////////////////////////////
/// HdrDataVisitor
////////////////////////////////////////////////////////////////////////////////

class HdrDataDeserializer : public ISubTimeFrameVisitor {
public:
  HdrDataDeserializer(O2Device& pDevice, const std::string& pChan, const int pChanId)
    : mDevice{ pDevice }, mChan{ pChan }, mChanId{ pChanId }
  {
  }

  void visit(O2SubTimeFrameLinkData& pStfLinkData) override;
  void visit(O2SubTimeFrameCruData& pStfCruData) override;
  void visit(O2SubTimeFrameRawData& pStfRawData) override;
  void visit(O2SubTimeFrame& pStf) override;

  bool deserialize(O2SubTimeFrame&);

  std::deque<FairMQMessagePtr> mHeaderMessages;
  std::deque<FairMQMessagePtr> mDataMessages;

private:
  O2Device& mDevice;
  const std::string& mChan;
  const int mChanId;
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_VISITORS_H_ */
