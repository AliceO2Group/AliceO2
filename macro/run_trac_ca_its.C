#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>

#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>
#include <FairMCEventHeader.h>

#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"

#include "ITStracking/Event.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"

#include "MathUtils/Utils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

void run_trac_ca_its(std::string path = "./", std::string outputfile = "o2ca_its.root",
                     std::string inputClustersITS = "o2clus_its.root", std::string inputGeom = "O2geometry.root",
                     std::string inputGRP = "o2sim_grp.root", std::string simfilename = "o2sim.root",
                     std::string paramfilename = "o2sim_par.root")
{

  o2::ITS::CA::Tracker<false> tracker;
  o2::ITS::CA::Event event;

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object" << FairLogger::endl;
  }
  o2::Base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  o2::Base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load ma" << FairLogger::endl;
  }
  double origD[3] = { 0., 0., 0. };
  tracker.setBz(field->getBz(origD));

  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute" << FairLogger::endl;
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode" << FairLogger::endl;

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
  std::uint32_t roFrame = 0;
  for (int iEvent = 0; iEvent < itsClusters.GetEntries(); ++iEvent) {
    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);

    if (isContITS) {
      int nclLeft = clusters->size();
      while (nclLeft > 0) {
        int nclUsed = o2::ITS::CA::IOUtils::loadROFrameData(roFrame, event, clusters, labels);
        if (nclUsed) {
          cout << "Event " << iEvent << " ROFrame " << roFrame << std::endl;
          // Attention: in the continuous mode cluster entry ID does not give the physics event ID
          // so until we use real vertexer, we have to work with vertex at origin
          event.addPrimaryVertex(0.f, 0.f, 0.f);
          tracker.setROFrame(roFrame);
          tracker.clustersToTracks(event);
          tracksITS->swap(tracker.getTracks());
          *trackLabels = tracker.getTrackLabels(); /// FIXME: assignment ctor is not optimal.
          outTree.Fill();
          nclLeft -= nclUsed;
        }
        roFrame++;
      }
    } else { // triggered mode
      cout << "Event " << iEvent << std::endl;
      o2::ITS::CA::IOUtils::loadEventData(event, clusters, labels);
      event.addPrimaryVertex(mcHeader->GetX(), mcHeader->GetY(), mcHeader->GetZ());
      tracker.clustersToTracks(event);
      tracksITS->swap(tracker.getTracks());
      *trackLabels = tracker.getTrackLabels(); /// FIXME: assignment ctor is not optimal.
      outTree.Fill();
    }
  }
  outFile.cd();
  outTree.Write();
  outFile.Close();
}

#endif
