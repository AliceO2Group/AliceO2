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

#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/VertexerBase.h"

#include "MathUtils/Utils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUChainITS.h"
using namespace o2::gpu;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void run_trac_ca_its(bool useITSVertex = false,
                     std::string path = "./",
                     std::string outputfile = "o2ca_its.root",
                     std::string inputClustersITS = "o2clus_its.root", std::string inputGeom = "O2geometry.root",
                     std::string inputGRP = "o2sim_grp.root", std::string simfilename = "o2sim.root",
                     std::string paramfilename = "o2sim_par.root")
{

  gSystem->Load("libITStracking.so");

  std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance());
  auto* chainTracking = rec->AddChain<GPUChainTracking>();
  auto* chainITS = rec->AddChain<GPUChainITS>();
  rec->Init();

  o2::ITS::Tracker tracker(chainITS->GetITSTrackerTraits());
  //o2::ITS::Tracker tracker(new o2::ITS::TrackerTraitsCPU());
  o2::ITS::ROframe event(0);

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(FATAL) << "Cannot run w/o GRP object" << FairLogger::endl;
  }
  o2::base::GeometryManager::loadGeometry(path + inputGeom, "FAIRGeom");
  o2::base::Propagator::initFieldFromGRP(grp);
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
  std::vector<o2::itsmft::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

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
  //  outTree.Branch("EventHeader.", &header);
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
        int nclUsed = o2::ITS::IOUtils::loadROFrameData(roFrame, event, clusters, labels);
        if (nclUsed) {
          cout << "Event " << iEvent << " ROFrame " << roFrame << std::endl;
          // Attention: in the continuous mode cluster entry ID does not give the physics event ID
          // so until we use real vertexer, we have to work with vertex at origin
          if (useITSVertex) {
            o2::ITS::VertexerBase vertexer(event);
            vertexer.setROFrame(roFrame);
            vertexer.initialise({ 0.005, 0.002, 0.04, 0.8, 5 });
            // set to true to use MC check
            vertexer.findTracklets(false);
            vertexer.findVertices();
            std::vector<Vertex> vertITS = vertexer.getVertices();
            if (!vertITS.empty()) {
              // Using only the first vertex in the list
              cout << " - Reconstructed vertexer: x = " << vertITS[0].getX() << " y = " << vertITS[0].getY() << " x = " << vertITS[0].getZ() << std::endl;
              event.addPrimaryVertex(vertITS[0].getX(), vertITS[0].getY(), vertITS[0].getZ());
            } else {
              cout << " - Vertex not reconstructed, tracking skipped" << std::endl;
              ;
            }
          } else {
            event.addPrimaryVertex(0.f, 0.f, 0.f);
          }
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
      o2::ITS::IOUtils::loadEventData(event, clusters, labels);
      if (useITSVertex) {
        o2::ITS::VertexerBase vertexer(event);
        vertexer.setROFrame(roFrame);
        vertexer.initialise({ 0.005, 0.002, 0.04, 0.8, 5 });
        // set to true to use MC check
        vertexer.findTracklets(false);
        vertexer.findVertices();
        std::vector<Vertex> vertITS = vertexer.getVertices();
        // Using only the first vertex in the list
        if (!vertITS.empty()) {
          cout << " - Reconstructed vertex: x = " << vertITS[0].getX() << " y = " << vertITS[0].getY() << " x = " << vertITS[0].getZ() << std::endl;
          event.addPrimaryVertex(vertITS[0].getX(), vertITS[0].getY(), vertITS[0].getZ());
        } else {
          cout << " - Vertex not reconstructed, tracking skipped" << std::endl;
        }
      } else {
        event.addPrimaryVertex(mcHeader->GetX(), mcHeader->GetY(), mcHeader->GetZ());
      }
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
