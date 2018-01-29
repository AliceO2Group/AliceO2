// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Helpers.cxx
/// \author David Rohr

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCBase/Constants.h"

using namespace o2::DataFormat::TPC;
using namespace o2::TPC;

std::unique_ptr<ClusterNativeAccessFullTPC> TPCClusterFormatHelper::accessNativeContainerArray(std::vector<ClusterNativeContainer>& clusters, std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* mcTruth)
{
  std::unique_ptr<ClusterNativeAccessFullTPC> retVal(new ClusterNativeAccessFullTPC);
  memset(retVal.get(), 0, sizeof(*retVal));
  for (int i = 0;i < clusters.size();i++)
  {
    retVal->clusters[clusters[i].sector][clusters[i].globalPadRow] = clusters[i].clusters.data();
    retVal->nClusters[clusters[i].sector][clusters[i].globalPadRow] = clusters[i].clusters.size();
    if (mcTruth) retVal->clustersMCTruth[clusters[i].sector][clusters[i].globalPadRow] = &(*mcTruth)[i];
  }
  return(std::move(retVal));
}
