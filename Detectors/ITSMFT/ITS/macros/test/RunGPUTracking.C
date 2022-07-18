#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "ITStracking/Vertexer.h"
#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>
#include <FairEventHeader.h>
#include <FairGeoParSet.h>
#include <FairLogger.h>
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "SimulationDataFormat/MCEventHeader.h"

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/Tracker.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#include "MathUtils/Utils.h"
#include "DetectorsBase/Propagator.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ITStracking/Configuration.h"

using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void RunGPUTracking(bool useLUT = true,
                    std::string path = "./",
                    std::string outputfile = "o2trac_its.root",
                    std::string inputClustersITS = "o2clus_its.root",
                    std::string matLUTFile = "matbud.root",
                    std::string inputGRP = "o2sim_grp.root",
                    long timestamp = 0)
{
  o2::its::ROframe event(0, 7);

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(fatal) << "Cannot run w/o GRP object";
  }

  o2::base::GeometryManager::loadGeometry(path);
  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(fatal) << "Failed to load ma";
  }
  double origD[3] = {0., 0., 0.};
  // tracker.setBz(field->getBz(origD));

  //-------- init lookuptable --------//
  if (useLUT) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
  } else {
    // tracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrTGeo);
  }

  // if (tracker.isMatLUT()) {
  //   LOG(info) << "Loaded material LUT from " << matLUTFile;
  // } else {
  //   LOG(info) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
  // }

  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms

  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL("http://alice-ccdb.cern.ch");
  mgr.setTimestamp(timestamp ? timestamp : o2::ccdb::getCurrentTimestamp());
  const o2::itsmft::TopologyDictionary* dict = mgr.get<o2::itsmft::TopologyDictionary>("ITS/Calib/ClusterDictionary");

  //>>>---------- attach input data --------------->>>
  TChain itsClusters("o2sim");
  itsClusters.AddFile((path + inputClustersITS).data());

  if (!itsClusters.GetBranch("ITSClusterComp")) {
    LOG(fatal) << "Did not find ITS clusters branch ITSClusterComp in the input tree";
  }
  std::vector<o2::itsmft::CompClusterExt>* cclusters = nullptr;
  itsClusters.SetBranchAddress("ITSClusterComp", &cclusters);

  if (!itsClusters.GetBranch("ITSClusterPatt")) {
    LOG(fatal) << "Did not find ITS cluster patterns branch ITSClusterPatt in the input tree";
  }
  std::vector<unsigned char>* patterns = nullptr;
  itsClusters.SetBranchAddress("ITSClusterPatt", &patterns);

  MCLabCont* labels = nullptr;
  if (!itsClusters.GetBranch("ITSClusterMCTruth")) {
    LOG(warning) << "Did not find ITS clusters branch ITSClusterMCTruth in the input tree";
  } else {
    itsClusters.SetBranchAddress("ITSClusterMCTruth", &labels);
  }

  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(fatal) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }

  std::vector<o2::itsmft::MC2ROFRecord>* mc2rofs = nullptr;
  if (!itsClusters.GetBranch("ITSClustersMC2ROF")) {
    LOG(fatal) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }
  itsClusters.SetBranchAddress("ITSClustersMC2ROF", &mc2rofs);

  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClusters.SetBranchAddress("ITSClustersROF", &rofs);
  itsClusters.GetEntry(0);

  //-------------------------------------------------
  std::unique_ptr<o2::gpu::GPUReconstruction> recCUDA(o2::gpu::GPUReconstruction::CreateInstance(o2::gpu::GPUDataTypes::DeviceType::CUDA, true));
  auto* chainITSCUDA = recCUDA->AddChain<o2::gpu::GPUChainITS>();
  std::unique_ptr<o2::its::Vertexer> vertexerCUDA = std::make_unique<o2::its::Vertexer>(chainITSCUDA->GetITSVertexerTraits());
  std::unique_ptr<o2::its::Tracker> trackerCUDA = std::make_unique<o2::its::Tracker>(chainITSCUDA->GetITSTrackerTraits());

  std::unique_ptr<o2::gpu::GPUReconstruction> recHIP(o2::gpu::GPUReconstruction::CreateInstance(o2::gpu::GPUDataTypes::DeviceType::HIP, true));
  auto* chainITSHIP = recHIP->AddChain<o2::gpu::GPUChainITS>();
  std::unique_ptr<o2::its::Vertexer> vertexerHIP = std::make_unique<o2::its::Vertexer>(chainITSHIP->GetITSVertexerTraits());
  std::unique_ptr<o2::its::Tracker> trackerHIP = std::make_unique<o2::its::Tracker>(chainITSHIP->GetITSTrackerTraits());

  o2::its::VertexingParameters parameters;
  parameters.phiCut = 0.005f;
  parameters.tanLambdaCut = 0.002f;

  vertexerCUDA->setParameters(parameters);
  vertexerHIP->setParameters(parameters);

  gsl::span<const unsigned char> patt(patterns->data(), patterns->size());
  auto pattIt = patt.begin();
  auto clSpan = gsl::span(cclusters->data(), cclusters->size());

  for (auto& rof : *rofs) {

    auto it = pattIt;
    o2::its::ioutils::loadROFrameData(rof, event, clSpan, pattIt, dict, labels);

    //   // CUDA
    vertexerCUDA->initialiseVertexer(&event);
    vertexerCUDA->findTracklets();
    // vertexerCUDA.filterMCTracklets(); // to use MC check
    vertexerCUDA->validateTracklets();
    vertexerCUDA->findVertices();
    std::vector<Vertex> vertITSCU = vertexerCUDA->exportVertices();
    if (!vertITSCU.empty()) {
      std::cout << " - Reconstructed vertex: x = " << vertITSCU[0].getX() << " y = " << vertITSCU[0].getY() << " x = " << vertITSCU[0].getZ() << std::endl;
      event.addPrimaryVertex(vertITSCU[0].getX(), vertITSCU[0].getY(), vertITSCU[0].getZ());
    } else {
      std::cout << " - Vertex not reconstructed" << std::endl;
    }

    //   // HIP
    vertexerHIP->initialiseVertexer(&event);
    vertexerHIP->findTracklets();
    //   // vertexerHIP.filterMCTracklets(); // to use MC check
    vertexerHIP->validateTracklets();
    vertexerHIP->findVertices();
    std::vector<Vertex> vertITSHIP = vertexerHIP->exportVertices();
    if (!vertITSHIP.empty()) {
      std::cout << " - Reconstructed vertex: x = " << vertITSHIP[0].getX() << " y = " << vertITSHIP[0].getY() << " x = " << vertITSHIP[0].getZ() << std::endl;
      event.addPrimaryVertex(vertITSHIP[0].getX(), vertITSHIP[0].getY(), vertITSHIP[0].getZ());
    } else {
      std::cout << " - Vertex not reconstructed" << std::endl;
    }
  }
}
#endif
