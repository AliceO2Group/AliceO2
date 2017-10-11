// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/ReadoutDataModel.h"
#include "Common/SubTimeFrameDataModel.h"

#include <map>
#include <iterator>
#include <algorithm>

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrame
////////////////////////////////////////////////////////////////////////////////
SubTimeFrame::SubTimeFrame(uint64_t pStfId)
  : mHeader{ pStfId }
{
}

std::uint64_t SubTimeFrame::getDataSize() const
{
  std::uint64_t lDataSize = 0;

  for (const auto& lDataIdentMapIter : mData) {
    for (const auto& lSubSpecMapIter : lDataIdentMapIter.second) {
      for (const auto& lStfDataIter : lSubSpecMapIter.second) {
        lDataSize += lStfDataIter.mData->GetSize();
      }
    }
  }

  return lDataSize;
}

std::vector<EquipmentIdentifier> SubTimeFrame::getEquipmentIdentifiers() const
{
  std::vector<EquipmentIdentifier> lKeys;

  for (const auto& lDataIdentMapIter : mData) {
    for (const auto& lSubSpecMapIter : lDataIdentMapIter.second) {
      lKeys.emplace_back(EquipmentIdentifier(lDataIdentMapIter.first, lSubSpecMapIter.first));
    }
  }

  return lKeys;
}

// TODO: make sure to report miss-configured equipment specs
static bool fixme__EqupId = true;

void SubTimeFrame::mergeStf(std::unique_ptr<SubTimeFrame> pStf)
{
  // make sure data equipment does not repeat
  std::set<EquipmentIdentifier> lUnionSet;
  for (const auto& lId : getEquipmentIdentifiers())
    lUnionSet.emplace(lId);

  for (const auto& lId : pStf->getEquipmentIdentifiers()) {
    if (lUnionSet.emplace(lId).second == false /* not inserted */) {
      LOG(WARNING) << "Equipment already present" << lId.info();
      if (fixme__EqupId) {
        LOG(WARNING) << "FIXME: should not continue";
        continue;
      } else {
        throw std::invalid_argument("Cannot add Equipment: already present!");
      }
    }
  }

  // merge the Stfs
  for (auto& lDataIdentMapIter : pStf->mData) {
    for (auto& lSubSpecMapIter : lDataIdentMapIter.second) {
      // source
      const DataIdentifier& lDataId = lDataIdentMapIter.first;
      const DataHeader::SubSpecificationType& lSubSpec = lSubSpecMapIter.first;
      StfDataVector& lStfDataVec = lSubSpecMapIter.second;

      // destination
      StfDataVector& lDstStfDataVec = mData[lDataId][lSubSpec];

      std::move(
        lStfDataVec.begin(),
        lStfDataVec.end(),
        std::back_inserter(lDstStfDataVec));
    }
  }

  // delete pStf
  pStf.reset();
}
}
} /* o2::DataDistribution */
