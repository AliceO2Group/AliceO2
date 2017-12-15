// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/SubTimeFrameUtils.h"
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
/// DataOriginSplitter
////////////////////////////////////////////////////////////////////////////////
void DataIdentifierSplitter::visit(SubTimeFrameDataSource& pStf)
{
  /* identifier filtering only */
}

void DataIdentifierSplitter::visit(O2SubTimeFrame& pStf)
{
  if (mDataIdentifier.dataOrigin == gDataOriginAny) {
    mSubTimeFrame = std::move(pStf);
  } else if (mDataIdentifier.dataDescription == gDataDescriptionAny) {
    std::vector<DataIdentifier> lToErase;
    // filter any source with requested origin
    for (auto &lKeyData : pStf.mStfReadoutData) {
      const DataIdentifier& lIden = lKeyData.first;
      if (lIden.dataOrigin == mDataIdentifier.dataOrigin) {
        // use data identifier of the object
        mSubTimeFrame.mStfReadoutData[lIden] = std::move(lKeyData.second);
        lToErase.emplace_back(lIden);
      }
    }
    // erase forked elements
    for (auto &lIden : lToErase)
      pStf.mStfReadoutData.erase(lIden);

  } else if (pStf.mStfReadoutData.count(mDataIdentifier) == 1) {
    mSubTimeFrame.mStfReadoutData[mDataIdentifier] = std::move(pStf.mStfReadoutData[mDataIdentifier]);
    pStf.mStfReadoutData.erase(mDataIdentifier);
  }

  // update element counts
  pStf.mStfHeader->payloadSize = pStf.mStfReadoutData.size();
  mSubTimeFrame.mStfHeader->payloadSize = mSubTimeFrame.mStfReadoutData.size();
}


O2SubTimeFrame DataIdentifierSplitter::split(O2SubTimeFrame& pStf, const DataIdentifier& pDataIdent, const int pChanId)
{
  mSubTimeFrame = O2SubTimeFrame(pChanId, pStf.Header().mStfId);
  mDataIdentifier = pDataIdent;

  pStf.accept(*this);

  return std::move(mSubTimeFrame);
}

}
} /* o2::DataDistribution */
