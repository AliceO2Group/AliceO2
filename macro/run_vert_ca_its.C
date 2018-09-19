#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <string>
#include <array>
#include <chrono>
#include <vector>
#include <cmath>
#include <sstream>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TGeoGlobalMagField.h>
#include <TH1I.h>

#include <FairLogger.h>
#include "FairRunAna.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairGeoParSet.h"
#include "FairMCEventHeader.h"

#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsITSMFT/Cluster.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/Event.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/ClusterLines.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using o2::ITS::CA::Cluster;
using o2::ITS::CA::Line;
using o2::ITS::CA::MathUtils::calculatePhiCoordinate;
using o2::ITS::CA::MathUtils::calculateRCoordinate;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void run_vert_ca_its(const int inspEvt = -1, bool useMC = false,
                     std::tuple<float, float, float, float, int> initParams = { 0.005, 0.002, 0.04, 0.8, 5 },
                     std::string path = "./", std::string inputClustersITS = "o2clus_its.root",
                     std::string inputGeom = "O2geometry.root", std::string inputGRP = "o2sim_grp.root",
                     std::string paramfilename = "o2sim_par.root", std::string simfilename = "o2sim.root",
                     std::string outfile = "vertexer_data.root")
{

  o2::ITS::CA::Event event;
  if (path.back() != '/')
    path += '/';
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
  // tracker.setBz(field->getBz(origD));
  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readoute" << FairLogger::endl;
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(INFO) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode" << FairLogger::endl;

  // Setup Runtime DB
  TFile paramFile(paramfilename.data());
  paramFile.Get("FairGeoParSet");
  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  //>>>---------- attach input data --------------->>>
  TChain mcHeaderTree("o2sim");
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());
  mcHeaderTree.AddFile((path + simfilename).data());

  //<<<---------- attach input data ---------------<<<
  if (!itsClusters.GetBranch("ITSCluster"))
    LOG(FATAL) << "Did not find ITS clusters branch ITSCluster in the input tree" << FairLogger::endl;

  std::vector<o2::ITSMFT::Cluster>* clusters = nullptr;
  itsClusters.SetBranchAddress("ITSCluster", &clusters);

  if (!itsClusters.GetBranch("ITSClusterMCTruth"))
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree" << FairLogger::endl;

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);

  FairMCEventHeader* header = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader."))
    LOG(FATAL) << "Did not find MC event header in the input header file." << FairLogger::endl;

  mcHeaderTree.SetBranchAddress("MCEventHeader.", &header);
  TFile* outputfile = new TFile(outfile.data(), "recreate");
  // Output tree
  TTree outTree("o2sim", "Vertexer Vertices");
  std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>* verticesITS =
    new std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>;
  outTree.Branch("ITSVertices", &verticesITS);

  // profiling
  std::chrono::time_point<std::chrono::system_clock> start, end_inst, end_init, end_track, end_vert;
  int instance_time, initiali_time, tracking_time = 0, vertexin_time = 0;

  TH1I timet("trackleting_duration", "trackleting_duration", 100, 0, 4000);
  TH1I timev("vertexing_duration", "vertexing_duration", 100, 0, 4000);
  std::uint32_t roFrame = 0;

  const int stopAt = (inspEvt == -1) ? itsClusters.GetEntries() : inspEvt + 1;
  for (int iEvent = (inspEvt == -1) ? 0 : inspEvt; iEvent < stopAt; ++iEvent) {
    int idx{ 0 };
    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);

    if (isContITS) {
      int nclLeft = clusters->size();
      while (nclLeft > 0) {
        int nclUsed = o2::ITS::CA::IOUtils::loadROFrameData(roFrame, event, clusters, labels);
        if (nclUsed) {
          cout << "Event " << iEvent << " ROFrame " << roFrame << std::endl;
          start = std::chrono::system_clock::now();
          o2::ITS::CA::Vertexer vertexer(event);
          end_inst = std::chrono::system_clock::now();
          vertexer.setROFrame(roFrame);
          vertexer.initialise(initParams);
          end_init = std::chrono::system_clock::now();
          vertexer.findTracklets(useMC);
          end_track = std::chrono::system_clock::now();
          vertexer.findVertices();
          end_vert = std::chrono::system_clock::now();
          std::vector<Vertex> vertITS = vertexer.getVertices();
          verticesITS->swap(vertITS);
          instance_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_inst - start).count();
          initiali_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start).count();
          tracking_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_track - start).count();
          vertexin_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_vert - start).count();
          std::cout << "\tInstance elapsed time: " << instance_time << "ms\n";
          std::cout << "\tInitialisation   time: " << initiali_time << "ms\n";
          std::cout << "\tTrackleting      time: " << tracking_time << "ms\n";
          std::cout << "\tVertex finding   time: " << vertexin_time << "ms\n";
          std::cout << "\tTotal: " << instance_time + initiali_time + tracking_time + vertexin_time << "ms\n";
          std::cout << "\ttracklets found: " << vertexer.getTracklets().size() << std::endl;
          std::cout << "\tvertices found: " << verticesITS->size() << std::endl;
          nclLeft -= nclUsed;
        }
        roFrame++;
        outTree.Fill();
      }
    } else { // triggered mode
      cout << "Event " << iEvent << std::endl;
      o2::ITS::CA::IOUtils::loadEventData(event, clusters, labels);
      start = std::chrono::system_clock::now();
      o2::ITS::CA::Vertexer vertexer(event);
      end_inst = std::chrono::system_clock::now();
      vertexer.setROFrame(roFrame);
      vertexer.initialise(initParams);
      end_init = std::chrono::system_clock::now();
      vertexer.findTracklets(useMC);
      end_track = std::chrono::system_clock::now();
      vertexer.findVertices();
      end_vert = std::chrono::system_clock::now();
      std::vector<Vertex> vertITS = vertexer.getVertices();
      verticesITS->swap(vertITS);
      instance_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_inst - start).count();
      initiali_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start).count();
      tracking_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_track - start).count();
      vertexin_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_vert - start).count();
      std::cout << "\tInstance elapsed time: " << instance_time << "ms\n";
      std::cout << "\tInitialisation   time: " << initiali_time << "ms\n";
      std::cout << "\tTrackleting      time: " << tracking_time << "ms\n";
      std::cout << "\tVertex finding   time: " << vertexin_time << "ms\n";
      std::cout << "\tTotal: " << instance_time + initiali_time + tracking_time + vertexin_time << "ms\n";
      std::cout << "\ttracklets found: " << vertexer.getTracklets().size() << std::endl;
      std::cout << "\tvertices found: " << verticesITS->size() << std::endl;
      outTree.Fill();
    }
    timet.Fill(tracking_time);
    timev.Fill(vertexin_time);
  } // loop on events
  timet.Write();
  timev.Write();
  outTree.Write();
  // verTupleResiduals->Write();
  // evtDumpFromVtxer->Write();
  outputfile->Close();
}
#endif