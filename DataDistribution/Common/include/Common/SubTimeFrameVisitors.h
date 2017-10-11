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

#include <vector>

class O2Device;
class FairMQChannel;

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameReadoutBuilder
////////////////////////////////////////////////////////////////////////////////

class SubTimeFrameReadoutBuilder : public ISubTimeFrameVisitor
{
 public:
  SubTimeFrameReadoutBuilder() = delete;
  SubTimeFrameReadoutBuilder(FairMQChannel& pChan);

  void addHbFrames(const ReadoutSubTimeframeHeader& pHdr, std::vector<FairMQMessagePtr>&& pHbFrames);
  std::unique_ptr<SubTimeFrame> getStf();

 protected:
  void visit(SubTimeFrame& pStf) override;

 private:
  std::unique_ptr<SubTimeFrame> mStf;
  FairMQChannel& mChan;
};

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataSerializer
////////////////////////////////////////////////////////////////////////////////

class InterleavedHdrDataSerializer : public ISubTimeFrameVisitor
{
 public:
  InterleavedHdrDataSerializer() = delete;
  InterleavedHdrDataSerializer(FairMQChannel& pChan)
    : mChan(pChan)
  {
    mMessages.reserve(1024);
  }

  void serialize(std::unique_ptr<SubTimeFrame>&& pStf);

 protected:
  void visit(SubTimeFrame& pStf) override;

 private:
  std::vector<FairMQMessagePtr> mMessages;
  FairMQChannel& mChan;
};

////////////////////////////////////////////////////////////////////////////////
/// InterleavedHdrDataDeserializer
////////////////////////////////////////////////////////////////////////////////

class InterleavedHdrDataDeserializer : public ISubTimeFrameVisitor
{
 public:
  InterleavedHdrDataDeserializer() = default;

  std::unique_ptr<SubTimeFrame> deserialize(FairMQChannel& pChan);
  std::unique_ptr<SubTimeFrame> deserialize(FairMQParts& pMsgs);

 protected:
  std::unique_ptr<SubTimeFrame> deserialize_impl();
  void visit(SubTimeFrame& pStf) override;

 private:
  std::vector<FairMQMessagePtr> mMessages;
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_VISITORS_H_ */
