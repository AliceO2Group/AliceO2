// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_DPL_H_
#define ALICEO2_SUBTIMEFRAME_DPL_H_

#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameVisitors.h"

#include <Headers/DataHeader.h>

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// StfDplAdapter
////////////////////////////////////////////////////////////////////////////////

class StfDplAdapter : public ISubTimeFrameVisitor
{
 public:
  StfDplAdapter() = delete;
  StfDplAdapter(FairMQChannel& pDplBridgeChan)
    : mSerializer(pDplBridgeChan)
  {
  }

  void sendToDpl(std::unique_ptr<SubTimeFrame>&& pStf)
  {
    mSerializer.serialize(std::move(pStf));
  }

 protected:
  void visit(SubTimeFrame& pStf) override{};

 private:
  InterleavedHdrDataSerializer mSerializer;
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_DPL_H_ */
