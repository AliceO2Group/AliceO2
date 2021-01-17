// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// EMCAL Correction Task
//
// Author: Raymond Ehlers

#include <cmath>

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"

#include "DetectorsBase/GeometryManager.h"

#include "AnalysisDataModel/EMCALClusters.h"

#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/Constants.h"
#include "DataFormatsEMCAL/AnalysisCluster.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/ClusterFactory.h"
#include "EMCALReconstruction/Clusterizer.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct EmcalCorrectionTask {
  Produces<o2::aod::EMCALClusters> clusters;

  // Options for the clusterization
  // 1 corresponds to EMCAL cells based on the Run2 definition.
  Configurable<int> selectedCellType{"selectedCellType", 1, "EMCAL Cell type"};
  Configurable<double> seedEnergy{"seedEnergy", 0.1, "Clusterizer seed energy."};
  Configurable<double> minCellEnergy{"minCellEnergy", 0.05, "Clusterizer minimum cell energy."};
  // TODO: Check this range, especially after change to the conversion...
  Configurable<double> timeCut{"timeCut", 10000, "Cell time cut"};
  Configurable<double> timeMin{"timeMin", 0, "Min cell time"};
  Configurable<double> timeMax{"timeMax", 10000, "Max cell time"};
  Configurable<bool> enableEnergyGradientCut{"enableEnergyGradientCut", true, "Enable energy gradient cut."};
  Configurable<double> gradientCut{"gradientCut", 0.03, "Clusterizer energy gradient cut."};

  // Clusterizer and related
  // Apparently streaming these objects really doesn't work, and causes problems for setting up the workflow.
  // So we use unique_ptr and define them below.
  std::unique_ptr<o2::emcal::Clusterizer<o2::emcal::Cell>> mClusterizer;
  std::unique_ptr<o2::emcal::ClusterFactory<o2::emcal::Cell>> mClusterFactory;
  // Cells and clusters
  std::vector<o2::emcal::Cell> mEmcalCells;
  std::vector<o2::emcal::AnalysisCluster> mAnalysisClusters;

  // QA
  // NOTE: This is not comprehensive.
  OutputObj<TH1F> hCellE{"hCellE"};
  OutputObj<TH1I> hCellTowerID{"hCellTowerID"};
  OutputObj<TH2F> hCellEtaPhi{"hCellEtaPhi"};
  OutputObj<TH2I> hCellRowCol{"hCellRowCol"};
  OutputObj<TH1F> hClusterE{"hClusterE"};
  OutputObj<TH2F> hClusterEtaPhi{"hClusterEtaPhi"};

  void init(InitContext const&)
  {
    LOG(DEBUG) << "Start init!";
    // NOTE: The geometry manager isn't necessary just to load the EMCAL geometry.
    //       However, it _is_ necessary for loading the misalignment matrices as of September 2020
    //       Eventually, those matrices will be moved to the CCDB, but it's not yet ready.
    // FIXME: Hardcoded for run 2
    o2::base::GeometryManager::loadGeometry(); // for generating full clusters
    LOG(DEBUG) << "After load geometry!";
    o2::emcal::Geometry* geometry = o2::emcal::Geometry::GetInstanceFromRunNumber(223409);
    if (!geometry) {
      LOG(ERROR) << "Failure accessing geometry";
    }

    // Setup clusterizer
    LOG(DEBUG) << "Init clusterizer!";
    mClusterizer = decltype(mClusterizer)(new o2::emcal::Clusterizer<o2::emcal::Cell>());
    mClusterizer->initialize(timeCut, timeMin, timeMax, gradientCut, enableEnergyGradientCut, seedEnergy, minCellEnergy);
    mClusterizer->setGeometry(geometry);
    LOG(DEBUG) << "Done with clusterizer. Setup cluster factory.";
    // Setup cluster factory.
    mClusterFactory = decltype(mClusterFactory)(new o2::emcal::ClusterFactory<o2::emcal::Cell>());
    LOG(DEBUG) << "Completed init!";

    // Setup QA hists.
    hCellE.setObject(new TH1F("hCellE", "hCellE", 200, 0.0, 100));
    hCellTowerID.setObject(new TH1I("hCellTowerID", "hCellTowerID", 20000, 0, 20000));
    hCellEtaPhi.setObject(new TH2F("hCellEtaPhi", "hCellEtaPhi", 160, -0.8, 0.8, 72, 0, 2 * 3.14159));
    // NOTE: Reversed column and row because it's more natural for presentatin.
    hCellRowCol.setObject(new TH2I("hCellRowCol", "hCellRowCol;Column;Row", 97, 0, 97, 600, 0, 600));
    hClusterE.setObject(new TH1F("hClusterE", "hClusterE", 200, 0.0, 100));
    hClusterEtaPhi.setObject(new TH2F("hClusterEtaPhi", "hClusterEtaPhi", 160, -0.8, 0.8, 72, 0, 2 * 3.14159));
  }

  //void process(aod::Collision const& collision, soa::Filtered<aod::Tracks> const& fullTracks, aod::Calos const& cells)
  //void process(aod::Collision const& collision, aod::Tracks const& tracks, aod::Calos const& cells)
  //void process(aod::BCs const& bcs, aod::Collision const& collision, aod::Calos const& cells)
  // Appears to need the BC to be accessed to be available in the collision table...
  void process(aod::Collision const& collision, aod::Calos const& cells, aod::BCs const& bcs)
  {
    LOG(DEBUG) << "Starting process.";
    // Convert aod::Calo to o2::emcal::Cell which can be used with the clusterizer.
    // In particular, we need to filter only EMCAL cells.
    mEmcalCells.clear();
    for (auto& cell : cells) {
      if (cell.caloType() != selectedCellType || cell.bc() != collision.bc()) {
        //LOG(DEBUG) << "Rejected";
        continue;
      }
      //LOG(DEBUG) << "Cell E: " << cell.getEnergy();
      //LOG(DEBUG) << "Cell E: " << cell;

      mEmcalCells.emplace_back(o2::emcal::Cell(
        cell.cellNumber(),
        cell.amplitude(),
        cell.time(),
        o2::emcal::intToChannelType(cell.cellType())));
    }

    // Cell QA
    // For convenience, use the clusterizer stored geometry to get the eta-phi
    for (auto& cell : mEmcalCells) {
      hCellE->Fill(cell.getEnergy());
      hCellTowerID->Fill(cell.getTower());
      auto res = mClusterizer->getGeometry()->EtaPhiFromIndex(cell.getTower());
      hCellEtaPhi->Fill(std::get<0>(res), std::get<1>(res));
      res = mClusterizer->getGeometry()->GlobalRowColFromIndex(cell.getTower());
      // NOTE: Reversed column and row because it's more natural for presentatin.
      hCellRowCol->Fill(std::get<1>(res), std::get<0>(res));
    }

    // TODO: Helpful for now, but should be removed.
    LOG(DEBUG) << "Converted EMCAL cells";
    for (auto& cell : mEmcalCells) {
      LOG(DEBUG) << cell.getTower() << ": E: " << cell.getEnergy() << ", time: " << cell.getTimeStamp() << ", type: " << cell.getType();
    }

    LOG(INFO) << "Converted cells. Contains: " << mEmcalCells.size() << ". Originally " << cells.size() << ". About to run clusterizer.";

    // Run the clusterizer
    mClusterizer->findClusters(mEmcalCells);
    LOG(DEBUG) << "Found clusters.";
    auto emcalClusters = mClusterizer->getFoundClusters();
    auto emcalClustersInputIndices = mClusterizer->getFoundClustersInputIndices();
    LOG(DEBUG) << "Retrieved results. About to setup cluster factory.";

    // Convert to analysis clusters.
    // First, the cluster factory requires cluster and cell information in order to build the clusters.
    mAnalysisClusters.clear();
    mClusterFactory->reset();
    mClusterFactory->setClustersContainer(*emcalClusters);
    mClusterFactory->setCellsContainer(mEmcalCells);
    mClusterFactory->setCellsIndicesContainer(*emcalClustersInputIndices);
    LOG(DEBUG) << "Cluster factory set up.";

    // Convert to analysis clusters.
    for (int icl = 0; icl < mClusterFactory->getNumberOfClusters(); icl++) {
      auto analysisCluster = mClusterFactory->buildCluster(icl);
      mAnalysisClusters.emplace_back(analysisCluster);
    }
    LOG(DEBUG) << "Converted to analysis clusters.";

    // Store the clusters in the table
    clusters.reserve(mAnalysisClusters.size());
    for (const auto& cluster : mAnalysisClusters) {
      // Determine the cluster eta, phi, correcting for the vertex position.
      auto pos = cluster.getGlobalPosition();
      pos = pos - math_utils::Point3D<float>{collision.posX(), collision.posY(), collision.posZ()};
      // Normalize the vector and rescale by energy.
      pos /= (cluster.E() / std::sqrt(pos.Mag2()));

      // We have our necessary properties. Now we store outputs
      //LOG(DEBUG) << "Cluster E: " << cluster.E();
      clusters(collision, cluster.E(), pos.Eta(), pos.Phi(), cluster.getM02());
      //if (cluster.E() < 0.300) {
      //    continue;
      //}
      hClusterE->Fill(cluster.E());
      hClusterEtaPhi->Fill(pos.Eta(), pos.Phi());
    }
    LOG(DEBUG) << "Done with process.";
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EmcalCorrectionTask>("emcal-correction-task")};
}
