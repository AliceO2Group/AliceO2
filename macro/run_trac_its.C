#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>

#include <TStopwatch.h>

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
#include "ITSReconstruction/CookedTracker.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif

using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

void run_trac_its(std::string path = "./", std::string outputfile = "o2trac_its.root",
                  std::string inputClustersITS = "o2clus_its.root", std::string inputGeom = "O2geometry.root",
                  std::string inputGRP = "o2sim_grp.root", std::string simfilename = "o2sim.root")
{

  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  // Setup timer
  TStopwatch timer;

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object" << FairLogger::endl;
  }
  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute" << FairLogger::endl;
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode" << FairLogger::endl;

  o2::Base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2GRot)); // request cached transforms

  o2::Base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(FATAL) << "Failed to load ma" << FairLogger::endl;
  }

  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  if (!itsClusters.GetBranch("ITSCluster")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree" << FairLogger::endl;
  }
  std::vector<o2::ITSMFT::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  MCLabCont* labels = nullptr;
  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(WARNING) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree" << FairLogger::endl;
  } else {
    itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);
  }
  //<<<---------- attach input data ---------------<<<

  //>>>--------- create/attach output ------------->>>
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "Cooked ITS Tracks");
  std::vector<o2::ITS::TrackITS>* tracksITS = new std::vector<o2::ITS::TrackITS>;
  MCLabCont* trackLabels = new MCLabCont();
  outTree.Branch("ITSTrack", &tracksITS);
  outTree.Branch("ITSTrackMCTruth", &trackLabels);
  //<<<--------- create/attach output -------------<<<

  //=================== INIT ==================
  Int_t n = 1;            // Number of threads
  Bool_t mcTruth = kTRUE; // kFALSE if no comparison with MC is needed
  o2::ITS::CookedTracker tracker(n);
  tracker.setContinuousMode(isContITS);
  tracker.setBz(field->solenoidField()); // in kG
  tracker.setGeometry(gman);
  tracker.setMCTruthContainers(labels, trackLabels);
  //===========================================

  //-------------------- settings -----------//
  std::uint32_t roFrame = 0;
  for (int iEvent = 0; iEvent < itsClusters.GetEntries(); ++iEvent) {

    std::vector<std::array<Double_t, 3>> vertices;
    vertices.emplace_back(std::array<Double_t, 3>{ 0., 0., 0. });

    itsClusters.GetEntry(iEvent);
    tracker.setVertices(vertices);
    tracker.process(*clusters, *tracksITS);
    outTree.Fill();
    tracksITS->clear();
    trackLabels->clear();
  }
  outFile.cd();
  outTree.Write();
  outFile.Close();

  timer.Stop();
  timer.Print();
}
