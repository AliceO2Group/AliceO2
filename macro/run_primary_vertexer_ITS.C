#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <memory>
#include <TChain.h>
#include <TFile.h>
#include <TSystem.h>
#include <TNtuple.h>

#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
// #include "ITStrackingCUDA/VertexerTraitsGPU.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"
#endif

using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using namespace o2::gpu;

int run_primary_vertexer_ITS(const GPUDataTypes::DeviceType dtype = GPUDataTypes::DeviceType::CPU,
                             const bool useMCcheck = false,
                             const int inspEvt = -1,
                             const int numEvents = 1,
                             const float phiCut = -1.f,
                             const float tanlambdaCut = -1.f,
                             const std::string inputClustersITS = "o2clus_its.root",
                             std::string dictfile = "",
                             const std::string inputGRP = "o2sim_grp.root",
                             const std::string simfilename = "o2sim_Kine.root",
                             const std::string path = "./")
{
  std::string gpuName;
  switch (dtype) {
    case GPUDataTypes::DeviceType::CUDA:
      // R__LOAD_LIBRARY(O2ITStrackingCUDA)
      gpuName = "vertexer_cuda";
      break;
    case GPUDataTypes::DeviceType::HIP:
      // R__LOAD_LIBRARY(O2ITStrackingHIP)
      gpuName = "vertexer_hip";
      break;
    default:
      gpuName = "vertexer_serial";
      break;
  }

  std::unique_ptr<GPUReconstruction> rec(GPUReconstruction::CreateInstance(dtype, true));
  auto* chainITS = rec->AddChain<GPUChainITS>();
  rec->Init();
  o2::its::Vertexer vertexer(chainITS->GetITSVertexerTraits());
  // o2::its::Vertexer vertexer(new o2::its::VertexerTraits());

  std::string mcCheck = useMCcheck ? "_data_MCCheck" : "_data";
  std::string outfile = gpuName + mcCheck + ".root";

  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  const bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  const bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  std::cout << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode" << std::endl;
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  o2::base::GeometryManager::loadGeometry(path);
  o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms

  // Get event header
  TChain mcHeaderTree("o2sim");
  mcHeaderTree.AddFile((path + simfilename).data());
  o2::dataformats::MCEventHeader* mcHeader = nullptr;
  if (!mcHeaderTree.GetBranch("MCEventHeader.")) {
    LOG(FATAL) << "Did not find MC event header in the input header file.";
  }
  mcHeaderTree.SetBranchAddress("MCEventHeader.", &mcHeader);

  itsClusters.AddFile((path + inputClustersITS).data());

  if (!itsClusters.GetBranch("ITSClusterComp")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClusterComp in the input tree";
  }
  std::vector<o2::itsmft::CompClusterExt>* cclusters = nullptr;
  itsClusters.SetBranchAddress("ITSClusterComp", &cclusters);

  if (!itsClusters.GetBranch("ITSClusterPatt")) {
    LOG(FATAL) << "Did not find ITS cluster patterns branch ITSClusterPatt in the input tree";
  }
  std::vector<unsigned char>* patterns = nullptr;
  itsClusters.SetBranchAddress("ITSClusterPatt", &patterns);

  MCLabCont* labels = nullptr;
  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(WARNING) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree";
  } else {
    itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);
  }

  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }

  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofs = nullptr;
  if (!itsClusters.GetBranch("ITSClustersMC2ROF")) {
    LOG(FATAL) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }
  itsClusters.SetBranchAddress("ITSClustersMC2ROF", &mc2rofs);

  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClusters.SetBranchAddress("ITSClustersROF", &rofs);

  itsClusters.GetEntry(0);
  //-------------------------------------------------

  o2::itsmft::TopologyDictionary dict;
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", "bin");
  }
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictfile.c_str();
    dict.readBinaryFile(dictfile);
  } else {
    LOG(INFO) << "Running without dictionary !";
  }

  //-------------------------------------------------

  TFile* outputfile = new TFile(outfile.data(), "recreate");

  TTree outTree("o2sim", "Vertexer Vertices");
  std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>* verticesITS =
    new std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>>;
  outTree.Branch("ITSVertices", &verticesITS);

  // benchmarks
  TNtuple foundVerticesBenchmark("foundVerticesBenchmark", "Found vertices benchmark", "frameId:foundVertices");
  TNtuple timeBenchmark("timeBenchmark", "Time benchmarks", "init:trackletFinder:trackletMatcher:vertexFinder:total");
  // \benchmarks

  std::uint32_t roFrame = 0;

  // Settings
  o2::its::VertexingParameters parameters;
  parameters.phiCut = phiCut > 0 ? phiCut : 0.005f;
  parameters.tanLambdaCut = tanlambdaCut > 0 ? tanlambdaCut : 0.002f;
  // e.g. parameters.clusterContributorsCut = 5;
  // \Settings

  const int stopAt = (inspEvt == -1) ? rofs->size() : inspEvt + numEvents;
  const int startAt = (inspEvt == -1) ? 0 : inspEvt;

  vertexer.setParameters(parameters);
  itsClusters.GetEntry(0);
  mcHeaderTree.GetEntry(0);

  gsl::span<const unsigned char> patt(patterns->data(), patterns->size());
  auto pattIt = patt.begin();
  auto clSpan = gsl::span(cclusters->data(), cclusters->size());

  for (size_t iROfCount{static_cast<size_t>(startAt)}; iROfCount < static_cast<size_t>(stopAt); ++iROfCount) {
    auto& rof = (*rofs)[iROfCount];
    o2::its::ROframe frame(iROfCount, 7); // to get meaningful roframeId
    std::cout << "ROframe: " << iROfCount << std::endl;

    auto it = pattIt;
    int nclUsed = o2::its::ioutils::loadROFrameData(rof, frame, clSpan, pattIt, dict, labels);

    std::array<float, 3> total{0.f, 0.f, 0.f};
    o2::its::ROframe* eventptr = &frame;

    // debug
    vertexer.setDebugTrackletSelection();
    // vertexer.setDebugLines(); // Handle with care, takes very long
    vertexer.setDebugCombinatorics();
    vertexer.setDebugSummaryLines();
    vertexer.setDebugCentroidsHistograms();
    // \debug

    auto logger = [](std::string s) { std::cout << s << std::endl; };
    total[0] = vertexer.evaluateTask(&o2::its::Vertexer::initialiseVertexer, "Vertexer initialisation", logger, eventptr);
    // total[1] = vertexer.evaluateTask(&o2::its::Vertexer::findTrivialMCTracklets, "Trivial Tracklet finding", std::cout); // If enable this, comment out the validateTracklets
    total[1] = vertexer.evaluateTask(&o2::its::Vertexer::findTracklets, "Tracklet finding", logger);
#ifdef _ALLOW_DEBUG_TREES_ITS_
    if (useMCcheck) {
      vertexer.evaluateTask(&o2::its::Vertexer::filterMCTracklets, "MC tracklets filtering", std::cout);
    }
#endif
    total[2] = vertexer.evaluateTask(&o2::its::Vertexer::validateTracklets, "Adjacent tracklets validation", logger);
    // In case willing to use the histogram-based CPU vertexer
    // total[3] = vertexer.evaluateTask(&o2::its::Vertexer::findHistVertices, "Vertex finding with histograms", std::cout);
    total[3] = vertexer.evaluateTask(&o2::its::Vertexer::findVertices, "Vertex finding", logger);

    std::vector<Vertex> vertITS = vertexer.exportVertices();
    const size_t numVert = vertITS.size();
    foundVerticesBenchmark.Fill(static_cast<float>(iROfCount), static_cast<float>(numVert));
    verticesITS->swap(vertITS);
    //   // TODO: get vertexer postion form MC truth
    //
    timeBenchmark.Fill(total[0], total[1], total[2], total[3], total[0] + total[1] + total[2] + total[3]);
    outTree.Fill();
  }

  outputfile->cd();
  outTree.Write();
  foundVerticesBenchmark.Write();
  timeBenchmark.Write();
  outputfile->Close();
  return 0;
}
