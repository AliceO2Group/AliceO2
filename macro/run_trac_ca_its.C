#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>
#include <FairMCEventHeader.h>

#include "DataFormatsITSMFT/Cluster.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"

#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/IOUtils.h"
#include "ITSReconstruction/CA/Tracker.h"

#include "MathUtils/Utils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

void run_trac_ca_its(std::string path = "./", std::string outputfile = "o2ca_its.root",
                     std::string inputClustersITS = "o2clus_its.root", std::string simfilename = "o2sim_mc.root",
                     std::string paramfilename = "o2sim_par.root")
{

  o2::ITS::CA::Tracker<false> tracker;
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

  // Get event header
  TChain mcHeaderTree("o2sim");
  mcHeaderTree.AddFile(simfilename.data());
  FairMCEventHeader* mcHeader = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader.")) {
    LOG(FATAL) << "Did not find MC event header in the input header file." << FairLogger::endl;
  }
  mcHeaderTree.SetBranchAddress("MCEventHeader.", &mcHeader);

  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  //<<<---------- attach input data ---------------<<<
  if (!itsClusters.GetBranch("ITSCluster")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree" << FairLogger::endl;
  }
  std::vector<o2::ITSMFT::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("EventHeader.")) {
    LOG(FATAL) << "Did not find the EventHeader branch in the input cluster tree" << FairLogger::endl;
  }
  FairEventHeader* header = nullptr;
  itsClusters.SetBranchAddress("EventHeader.", &header);

  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree" << FairLogger::endl;
  }
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "CA ITS Tracks");
  std::vector<o2::ITS::TrackITS>* tracksITS = new std::vector<o2::ITS::TrackITS>;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trackLabels =
    new o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  outTree.Branch("EventHeader.", &header);
  outTree.Branch("ITSTrack", &tracksITS);
  outTree.Branch("ITSTrackMCTruth", &trackLabels);

  //-------------------- settings -----------//
  for (int iEvent = 0; iEvent < itsClusters.GetEntries(); ++iEvent) {
    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);

    cout << "Event " << iEvent << std::endl;
    o2::ITS::CA::IOUtils::loadEventData(event, clusters, labels);
    event.addPrimaryVertex(mcHeader->GetX(), mcHeader->GetY(), mcHeader->GetZ());
    tracker.clustersToTracks(event);
    tracksITS->swap(tracker.getTracks());
    *trackLabels = tracker.getTrackLabels(); /// FIXME: assignment ctor is not optimal.
    outTree.Fill();
  }

  outFile.cd();
  outTree.Write();
  outFile.Close();
}

#endif
