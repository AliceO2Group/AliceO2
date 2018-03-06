#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>
#include <string>
#include <FairLogger.h>
#include "FairRunAna.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairGeoParSet.h"

#include "DataFormatsITSMFT/Cluster.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"

#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/IOUtils.h"
#include "ITSReconstruction/CA/vertexer/Vertexer.h"

#include "MathUtils/Utils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#endif

void run_vert_ca_its(std::string path = "./",
                // std::string outputfile = "o2ca_its.root",
                std::string inputClustersITS = "o2clus_its.root",
                std::string paramfilename = "o2sim_par.root")
{

  // o2::ITS::CA::Tracker<false> tracker;
  o2::ITS::CA::Event event;

  if (path.back() != '/') {
    path += '/';
  }

  // Setup Runtime DB
  TFile paramFile(paramfilename.data());
  paramFile.Get("FairGeoParSet");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  //<<<---------- attach input data ---------------<<<
  if (!itsClusters.GetBranch("ITSCluster")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree"
               << FairLogger::endl;
  }
  std::vector<o2::ITSMFT::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree"
               << FairLogger::endl;
  }
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> * labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  // create/attach output tree
  // TFile outFile((path + outputfile).data(), "recreate");
  // TTree outTree("catracks", "CA ITS Tracks");

  //-------------------- settings -----------//
  for (int iEvent = 0; iEvent < itsClusters.GetEntries(); ++iEvent) { // itsClusters.GetEntries()
    itsClusters.GetEntry(iEvent);
    o2::ITS::CA::IOUtils::loadEventData(event,clusters,labels);
    // event.addPrimaryVertex(0.,0.,0.);
    // tracker.clustersToTracks(event);
    o2::ITS::CA::Vertexer vertexer(event);
    vertexer.initialise(0.02, 0.005, 0.03, 0.8, 3);
    vertexer.findTracklets();
    // vertexer.computeTriplets();
    // vertexer.printIndexTables();
    // vertexer.checkTriplets();
    // vertexer.debugTracklets();
    vertexer.findVertices();
    vertexer.printVertices();
  }


  // outFile.cd();
  // outTree.Write();
  // outFile.Close();
}
