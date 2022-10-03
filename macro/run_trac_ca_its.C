#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

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
#include "DetectorsBase/Propagator.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"

#include "Field/MagneticField.h"

#include "ITSBase/GeometryTGeo.h"

#include "DataFormatsITSMFT/CompCluster.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/Vertexer.h"

#include "MathUtils/Utils.h"
#include "DetectorsBase/Propagator.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"

#include <TGraph.h>

#include "ITStracking/Configuration.h"

using namespace o2::gpu;
using o2::its::TrackingParameters;

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

void run_trac_ca_its(bool cosmics = false,
                     bool useLUT = true,
                     std::string path = "./",
                     std::string outputfile = "o2trac_its.root",
                     std::string inputClustersITS = "o2clus_its.root",
                     std::string matLUTFile = "matbud.root",
                     std::string inputGRP = "o2sim_grp.root",
                     long timestamp = 0)
{

  gSystem->Load("libO2ITStracking");

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

  //

  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(warning) << "ITS is not in the readout";
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(info) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode";

  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms

  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL("https://alice-ccdb.cern.ch");
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
  std::vector<o2::its::TrackITSExt> tracks;
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "CA ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  std::vector<o2::itsmft::ROFRecord> vertROFvec, *vertROFvecPtr = &vertROFvec;
  std::vector<Vertex> vertices, *verticesPtr = &vertices;

  std::vector<o2::MCCompLabel> trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);
  outTree.Branch("ITSTracksMC2ROF", &mc2rofs);
  outTree.Branch("Vertices", &verticesPtr);
  outTree.Branch("VerticesROF", &vertROFvecPtr);
  if (!itsClusters.GetBranch("ITSClustersROF")) {
    LOG(fatal) << "Did not find ITS clusters branch ITSClustersROF in the input tree";
  }

  o2::its::VertexerTraits* traits = new o2::its::VertexerTraits();
  o2::its::Vertexer vertexer(traits);

  o2::its::VertexingParameters parameters;
  parameters.phiCut = 0.005f;
  parameters.tanLambdaCut = 0.002f;
  vertexer.setParameters(parameters);

  int roFrameCounter{0};

  std::vector<double> ncls;
  std::vector<double> time;

  std::vector<TrackingParameters> trackParams(1);
  if (cosmics) {
    trackParams[0].MinTrackLength = 4;
    trackParams[0].CellDeltaTanLambdaSigma *= 400;
    trackParams[0].PhiBins = 4;
    trackParams[0].ZBins = 16;
    trackParams[0].PVres = 1.e5f;
    trackParams[0].FitIterationMaxChi2[0] = 1.e28;
    trackParams[0].FitIterationMaxChi2[1] = 1.e28;
  } else {
    // PbPb tracking params
    // ----
    // trackParams.resize(3);
    // trackParams[0].TrackletMaxDeltaPhi = 0.05f;
    // trackParams[0].DeltaROF = 0;
    // trackParams[1].CopyCuts(trackParams[0], 2.);
    // trackParams[1].DeltaROF = 0;
    // trackParams[2].CopyCuts(trackParams[1], 2.);
    // trackParams[2].DeltaROF = 1;
    // trackParams[2].MinTrackLength = 4;
    // ---
    // Uncomment for pp
    trackParams.resize(3);
    trackParams[1].TrackletMinPt = 0.2f;
    trackParams[2].TrackletMinPt = 0.1f;
    trackParams[2].DeltaROF = 1;
    trackParams[2].MinTrackLength = 4;
    // ---
  }

  int currentEvent = -1;
  gsl::span<const unsigned char> patt(patterns->data(), patterns->size());
  auto pattIt = patt.begin();
  auto pattIt_vertexer = patt.begin();
  auto clSpan = gsl::span(cclusters->data(), cclusters->size());

  o2::its::TimeFrame tf;
  gsl::span<o2::itsmft::ROFRecord> rofspan(*rofs);
  std::vector<bool> processingMask(rofs->size(), true);
  tf.loadROFrameData(rofspan, clSpan, pattIt_vertexer, dict, labels);
  tf.setMultiplicityCutMask(processingMask);

  int rofId{0};
  vertexer.adoptTimeFrame(tf);
  vertexer.clustersToVertices();

  tf.printVertices();

  o2::its::Tracker tracker(new o2::its::TrackerTraits);
  tracker.adoptTimeFrame(tf);

  if (useLUT) {
    auto* lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    tracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
  } else {
    tracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrTGeo);
  }

  if (tracker.isMatLUT()) {
    LOG(info) << "Loaded material LUT from " << matLUTFile;
  } else {
    LOG(info) << "Material LUT " << matLUTFile << " file is absent, only TGeo can be used";
  }

  tracker.setBz(field->getBz(origD));
  tracker.setParameters(trackParams);
  tracker.clustersToTracks();
  //-------- init lookuptable --------//

  for (int iROF{0}; iROF < tf.getNrof(); ++iROF) {
    tracksITS.clear();
    for (auto& trc : tf.getTracks(iROF)) {
      trc.setFirstClusterEntry(trackClIdx.size()); // before adding tracks, create final cluster indices
      int ncl = trc.getNumberOfClusters();
      for (int ic = 0; ic < ncl; ic++) {
        trackClIdx.push_back(trc.getClusterIndex(ic));
      }
      tracksITS.emplace_back(trc);
    }
    trackLabels = tf.getTracksLabel(iROF); /// FIXME: assignment ctor is not optimal.
    outTree.Fill();
  }

  outFile.cd();
  outTree.Write();
  // outFile.Close();

  // TGraph* graph = new TGraph(ncls.size(), ncls.data(), time.data());
  // graph->SetMarkerStyle(20);
  // graph->Draw("AP");
}

#endif
