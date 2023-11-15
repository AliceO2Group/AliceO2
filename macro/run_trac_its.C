// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
#include <FairMCEventHeader.h>
#include "Framework/Logger.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/CookedTracker.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

#include "ReconstructionDataFormats/PrimaryVertex.h" // hack to silence JIT compiler
#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"

using MCLabCont = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using MCLabContTr = std::vector<o2::MCCompLabel>;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

void run_trac_its(std::string path = "./", std::string outputfile = "o2trac_its.root",
                  std::string inputClustersITS = "o2clus_its.root",
                  std::string inputGeom = "",
                  std::string inputGRP = "o2sim_grp.root",
                  long timestamp = 0)
{

  // Setup timer
  TStopwatch timer;

  if (path.back() != '/') {
    path += '/';
  }

  //-------- init geometry and field --------//
  const auto grp = o2::parameters::GRPObject::loadFrom(path + inputGRP);
  if (!grp) {
    LOG(fatal) << "Cannot run w/o GRP object";
  }
  bool isITS = grp->isDetReadOut(o2::detectors::DetID::ITS);
  if (!isITS) {
    LOG(warning) << "ITS is not in the readoute";
    return;
  }
  bool isContITS = grp->isDetContinuousReadOut(o2::detectors::DetID::ITS);
  LOG(info) << "ITS is in " << (isContITS ? "CONTINUOS" : "TRIGGERED") << " readout mode";

  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot)); // request cached transforms

  o2::base::Propagator::initFieldFromGRP(grp);
  auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!field) {
    LOG(fatal) << "Failed to load ma";
  }

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
    LOG(warning) << "Did not find ITSClustersMC2ROF branch in the input tree";
  }
  itsClusters.SetBranchAddress("ITSClustersMC2ROF", &mc2rofs);

  std::vector<o2::itsmft::ROFRecord>* rofs = nullptr;
  itsClusters.SetBranchAddress("ITSClustersROF", &rofs);

  //>>>--------- create/attach output ------------->>>
  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("o2sim", "Cooked ITS Tracks");
  std::vector<o2::its::TrackITS> tracksITS, *tracksITSPtr = &tracksITS;
  std::vector<int> trackClIdx, *trackClIdxPtr = &trackClIdx;
  std::vector<o2::itsmft::ROFRecord> vertROFvec, *vertROFvecPtr = &vertROFvec;
  std::vector<Vertex> vertices, *verticesPtr = &vertices;

  MCLabContTr trackLabels, *trackLabelsPtr = &trackLabels;
  outTree.Branch("ITSTrack", &tracksITSPtr);
  outTree.Branch("ITSTrackClusIdx", &trackClIdxPtr);
  outTree.Branch("ITSTrackMCTruth", &trackLabelsPtr);
  outTree.Branch("ITSTracksROF", &rofs);
  outTree.Branch("ITSTracksMC2ROF", &mc2rofs);
  outTree.Branch("Vertices", &verticesPtr);
  outTree.Branch("VerticesROF", &vertROFvecPtr);
  //<<<--------- create/attach output -------------<<<

  //=================== INIT ==================
  Int_t n = 1;            // Number of threads
  Bool_t mcTruth = kTRUE; // kFALSE if no comparison with MC is needed
  o2::its::CookedTracker tracker(n);
  tracker.setContinuousMode(isContITS);
  tracker.setBz(field->solenoidField()); // in kG
  tracker.setGeometry(gman);
  if (mcTruth) {
    tracker.setMCTruthContainers(labels, trackLabelsPtr);
  }
  //===========================================

  o2::its::VertexerTraits vertexerTraits;
  o2::its::Vertexer vertexer(&vertexerTraits);

  int nTFs = itsClusters.GetEntries();
  for (int nt = 0; nt < nTFs; nt++) {
    LOGP(info, "Processing timeframe {}/{}", nt, nTFs);
    itsClusters.GetEntry(nt);
    o2::its::TimeFrame tf;
    gsl::span<o2::itsmft::ROFRecord> rofspan(*rofs);
    gsl::span<const unsigned char> patt(*patterns);

    auto pattIt = patt.begin();
    auto pattIt_vertexer = patt.begin();
    auto clSpan = gsl::span(cclusters->data(), cclusters->size());
    std::vector<bool> processingMask(rofs->size(), true);
    tf.loadROFrameData(rofspan, clSpan, pattIt_vertexer, dict, labels);
    tf.setMultiplicityCutMask(processingMask);
    vertexer.adoptTimeFrame(tf);
    vertexer.clustersToVertices();
    int iRof = 0;
    for (auto& rof : *rofs) {
      auto it = pattIt;

      auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
      vtxROF.setFirstEntry(vertices.size());       // dedicated ROFRecord
      std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>> verticesL;
      vtxROF.setNEntries(tf.getPrimaryVertices(iRof).size());

      for (const auto& vtx : tf.getPrimaryVertices(iRof)) {
        vertices.push_back(vtx);
        verticesL.push_back(vtx);
      }
      if (tf.getPrimaryVertices(iRof).empty()) {
        verticesL.emplace_back();
      }
      tracker.setVertices(verticesL);
      tracker.process(clSpan, it, dict, tracksITS, trackClIdx, rof);
      ++iRof;
    }
    outTree.Fill();
    if (mcTruth) {
      trackLabelsPtr->clear();
      mc2rofs->clear();
    }
    tracksITSPtr->clear();
    trackClIdxPtr->clear();
    rofs->clear();
    verticesPtr->clear();
    vertROFvecPtr->clear();
  }
  outFile.cd();
  outTree.Write();
  outFile.Close();

  timer.Stop();
  timer.Print();
}

#endif
