#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <string>
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"

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
#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"
using namespace o2::gpu;

R__LOAD_LIBRARY(libITStracking)
// R__LOAD_LIBRARY(libTPCReconstruction)
// gSystem->Load("libTPCReconstruction")

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

int run_test_vert_ca_its(const int inspEvt = -1,
                         std::string path = "./",
                         std::string inputClustersITS = "o2clus_its.root",
                         std::string inputGeom = "O2geometry.root",
                         std::string inputGRP = "o2sim_grp.root",
                         std::string paramfilename = "o2sim_par.root",
                         std::string simfilename = "o2sim.root",
                         std::string outfile = "vertexer_data.root")
{
  if (path.back() != '/')
    path += '/';
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
  // tracker.setBz(field->getBz(origD));
  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(WARNING) << "ITS is not in the readout" << FairLogger::endl;
    return -1;
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

  std::vector<o2::itsmft::Cluster>* clusters = nullptr;
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

  std::uint32_t roFrame = 0;

  const int stopAt = (inspEvt == -1) ? itsClusters.GetEntries() : inspEvt + 1;
  o2::ITS::ROframe frame(-123);

  // o2::ITS::Vertexer vertexer(GPUReconstruction::CreateInstance()->GetITSVerterTraits());
  GPUReconstruction* instance = GPUReconstruction::CreateInstance();
  GPUChainITS* chainITS = instance->AddChain<GPUChainITS>();
  o2::ITS::VertexerTraits* traits = chainITS->GetITSVertexerTraits();
  std::cout << "macro -> traits ptr -> " << traits << std::endl;
  // traits->dumpVertexerTraits();

  std::uint32_t ROframe{ 0 };
  // vertexer.setROframe(ROframe);
  for (int iEvent = (inspEvt == -1) ? 0 : inspEvt; iEvent < stopAt; ++iEvent) {
    itsClusters.GetEntry(iEvent);
    mcHeaderTree.GetEntry(iEvent);
    if (isContITS) {
      int nclLeft = clusters->size();
      int nclUsed = o2::ITS::IOUtils::loadROFrameData(roFrame, frame, clusters, labels);
      while (nclLeft > 0) {
        roFrame++;
      }
    } else {
      o2::ITS::IOUtils::loadEventData(frame, clusters, labels);
    }
    // vertexer.clustersToVertices(frame);
    // ROframe* frameptr = &frame;
    // vertexer.initialiseVertexer(&frame);
    // vertexer.dumpTraits();
    // std::vector<Vertex> vertITS = vertexer.exportVertices();
    // verticesITS->swap(vertITS);
    outTree.Fill();
  } // loop on events
  outTree.Write();
  outputfile->Close();
  return 0;
}
#endif
