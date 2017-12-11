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
  for (int i = 0;i < Constants::MAXSECTOR;i++)
  {
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++)
    {
      retVal->mClusters[i][j] = nullptr;
      retVal->mNClusters[i][j] = 0;
      retVal->mClustersMCTruth[i][j] = nullptr;
    }
  }  
  for (int i = 0;i < clusters.size();i++)
  {
    retVal->mClusters[clusters[i].mSector][clusters[i].mGlobalPadRow] = clusters[i].mClusters.data();
    retVal->mNClusters[clusters[i].mSector][clusters[i].mGlobalPadRow] = clusters[i].mClusters.size();
    if (mcTruth) retVal->mClustersMCTruth[clusters[i].mSector][clusters[i].mGlobalPadRow] = &(*mcTruth)[i];
  }
  return(std::move(retVal));
}
