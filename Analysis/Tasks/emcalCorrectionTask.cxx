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

#include "Analysis/EMCALClusters.h"

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
  //Produces<o2::aod::Jets> jets;
  //Produces<o2::aod::JetConstituents> constituents;

  // Options for the clusterization
  Configurable<int> selectedCellType{"selectedCellType", 1, "EMCAL Cell type"};

  //BkgMode bkgMode = BkgMode::none;
  //Configurable<double> rParamBkg{"rParamBkg", 0.2, "jet radius for background"};
  //Configurable<double> rapBkg{"rapBkg", .9, "rapidity range for background"};
  //// TODO: use configurables also for enums
  //fastjet::JetAlgorithm algorithmBkg{fastjet::kt_algorithm};
  //fastjet::RecombinationScheme recombSchemeBkg{fastjet::E_scheme};
  //fastjet::JetDefinition jetDefBkg{algorithmBkg, rParamBkg, recombSchemeBkg, strategy};
  //fastjet::AreaDefinition areaDefBkg{areaType, ghostSpec};
  //fastjet::Selector selBkg = fastjet::SelectorAbsRapMax(rapBkg);

  // Clusterizer and related
  // Apparently streaming these objects really doesn't work, and causes problems for setting up the workflow.
  // So we define them below.
  std::unique_ptr<o2::emcal::Clusterizer<o2::emcal::Cell>> mClusterizer;
  std::unique_ptr<o2::emcal::ClusterFactory<o2::emcal::Cell>> mClusterFactory;
  // Cells and clusters
  std::vector<o2::emcal::Cell> mEmcalCells;
  std::vector<o2::emcal::AnalysisCluster> mAnalysisClusters;

  //std::unique_ptr<fastjet::BackgroundEstimatorBase> bge;
  //std::unique_ptr<fastjet::Subtractor> sub;

  //std::vector<fastjet::PseudoJet> pJets;

  void init(InitContext const&)
  {
    LOG(INFO) << "Start init!";
    // Initialize clusterizer
    // FIXME: Placeholder configuration -> make configurable.
    double timeCut = 10000, timeMin = 0, timeMax = 10000, gradientCut = 0.03, thresholdSeedEnergy = 0.1, thresholdCellEnergy = 0.05;
    bool doEnergyGradientCut = true;

    // FIXME: Hardcoded for run 2
    // NOTE: The geometry manager isn't necessary just to load the EMCAL geometry.
    //       However, it _is_ necessary for loading the misalignment matrices as of September 2020
    //       Eventually, those matrices will be moved to the CCDB, but it's not yet ready.
    o2::base::GeometryManager::loadGeometry(); // for generating full clusters
    LOG(DEBUG) << "After load geometry!";
    o2::emcal::Geometry* geometry = o2::emcal::Geometry::GetInstanceFromRunNumber(223409);
    if (!geometry) {
      LOG(ERROR) << "Failure accessing geometry";
    }

    // Initialize clusterizer and link geometry
    LOG(INFO) << "Init clusterizer!";
    mClusterizer = decltype(mClusterizer)(new o2::emcal::Clusterizer<o2::emcal::Cell>());
    mClusterizer->initialize(timeCut, timeMin, timeMax, gradientCut, doEnergyGradientCut, thresholdSeedEnergy, thresholdCellEnergy);
    mClusterizer->setGeometry(geometry);
    LOG(INFO) << "Done with clusterizer. Setup cluster factory.";
    // Initialize cluster factory.
    mClusterFactory = decltype(mClusterFactory)(new o2::emcal::ClusterFactory<o2::emcal::Cell>());
    LOG(INFO) << "Completed init!";
  }

  //void process(aod::Collision const& collision, soa::Filtered<aod::Tracks> const& fullTracks, aod::Calos const& cells)
  //void process(aod::Collision const& collision, aod::Tracks const& tracks, aod::Calos const& cells)
  //void process(aod::BCs const& bcs, aod::Collision const& collision, aod::Calos const& cells)
  // Appears to need the BC to be accessed to be available in the collision table...
  void process(aod::Collision const& collision, aod::Calos const& cells, aod::BCs const& bcs)
  {
    LOG(INFO) << "Starting process.";
    // Convert aod::Calo to o2::emcal::Cell
    mEmcalCells.clear();
    for (auto & cell : cells) {
      // TODO: Select only EMCAL cells based on the CaloType
      //       Check in AliRoot.
      if (cell.caloType() != selectedCellType || cell.bc() != collision.bc()) {
        //LOG(INFO) << "Rejected";
        continue;
      }
      //LOG(INFO) << "Cell E: " << cell.getEnergy();
      //LOG(INFO) << "Cell E: " << cell;

      // TODO: It identifies almost all of the cells as low gain...? Seems like they should be high gain?
      mEmcalCells.emplace_back(o2::emcal::Cell(
        cell.cellNumber(),
        cell.amplitude(),
        cell.time(),
        o2::emcal::intToChannelType(cell.cellType())
      ));
    }

    LOG(INFO) << "Converted EMCAL cells";
    for (auto & cell : mEmcalCells) {
      LOG(INFO) << cell.getTower() <<  ": E: " << cell.getEnergy() << ", time: " << cell.getTimeStamp()  << ", type: " << cell.getType();
      //LOG(INFO) << cell;
    }

    LOG(INFO) << "Converted cells. Contains: " << mEmcalCells.size() << ". Originally " << cells.size() << ". About to run clusterizer.";

    // Run the clusterizer
    //mClusterizer->findClusters(gsl::span<o2::emcal::Cell>(mEmcalCells.begin(), mEmcalCells.size()));
    mClusterizer->findClusters(mEmcalCells);
    LOG(INFO) << "Found clusters.";
    auto emcalClusters = mClusterizer->getFoundClusters();
    auto emcalClustersInputIndices = mClusterizer->getFoundClustersInputIndices();
    LOG(INFO) << "Retrieved results. About to setup cluster factory.";

    // Convert to analysis clusters.
    mAnalysisClusters.clear();
    mClusterFactory->reset();
    mClusterFactory->setClustersContainer(*emcalClusters);
    mClusterFactory->setCellsContainer(mEmcalCells);
    mClusterFactory->setCellsIndicesContainer(*emcalClustersInputIndices);
    LOG(INFO) << "Cluster factory set up.";

    // Convert to analysis clusters.
    for (int icl = 0; icl < mClusterFactory->getNumberOfClusters(); icl++) {
      auto analysisCluster = mClusterFactory->buildCluster(icl);
      mAnalysisClusters.emplace_back(analysisCluster);
    }
    LOG(INFO) << "Converted to analysis clusters.";

    // Store the clusters in the table
    for (const auto& cluster : mAnalysisClusters) {
      // TODO: Determine eta, phi!
      //clusters(collision, cluster.E(), cluster.Eta(), cluster.Phi(), cluster.getM02());
      auto pos = cluster.getGlobalPosition();
      // Correct for the vertex position.
      pos = pos - Point3D<float>{collision.posX(), collision.posY(), collision.posZ()};
      // Normalize the vector and rescale by energy.
      pos /= (cluster.E() / std::sqrt(pos.Mag2()));
      //double px = ;
      //double py = ;
      //double pz = ;
      //double pt = std::sqrt(px * px + py * py);
      //double phi = atan2(py, px);
      //double eta = asinh(pz / pt);
      clusters(collision, cluster.E(), pos.Eta(), pos.Phi(), cluster.getM02());
      LOG(DEBUG) << "Cluster E: " << cluster.E();
    }
    LOG(INFO) << "Done with process.";
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EmcalCorrectionTask>("emcal-correction-task")};
}
