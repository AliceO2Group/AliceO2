// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#define MS_GSL_V3
#include "TFile.h"
#include "TTree.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#endif

using namespace o2::tpc;

void runClusters(std::string_view outputFile = "ClusterQC.root", std::string_view fileName = "tpc-native-clusters.root", const size_t maxNClusters = 0)
{
  ClusterNativeHelper::Reader tpcClusterReader;
  tpcClusterReader.init(fileName.data());

  ClusterNativeAccess clusterIndex;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;

  qc::Clusters clusters;

  for (int i = 0; i < tpcClusterReader.getTreeSize(); ++i) {
    std::cout << "Event " << i << "\n";
    tpcClusterReader.read(i);
    tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);
    size_t iClusters = 0;
    for (int isector = 0; isector < constants::MAXSECTOR; ++isector) {
      for (int irow = 0; irow < constants::MAXGLOBALPADROW; ++irow) {
        const int nClusters = clusterIndex.nClusters[isector][irow];
        if (!nClusters) {
          continue;
        }
        for (int icl = 0; icl < nClusters; ++icl) {
          const auto& cl = *(clusterIndex.clusters[isector][irow] + icl);
          clusters.processCluster(cl, Sector(isector), irow);
          ++iClusters;
          if (maxNClusters > 0 && iClusters >= maxNClusters) {
            return;
          }
        }
      }
    }
  }
  clusters.analyse();
  clusters.dumpToFile(outputFile.data());
}
