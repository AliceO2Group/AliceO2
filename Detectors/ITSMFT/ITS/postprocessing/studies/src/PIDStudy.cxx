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

/// \file PIDStudy.cxx
/// \brief Study to evaluate the PID performance of the ITS
/// \author Francesco Mazzaschi, fmazzasc@cern.ch

#include "ITSStudies/PIDStudy.h"
#include "ITSStudies/ITSStudiesConfigParam.h"

#include "Framework/Task.h"
#include "ITSBase/GeometryTGeo.h"
#include "Steer/MCKinematicsReader.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITStracking/IOUtils.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/DCA.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsTPC/PIDResponse.h"

#include <numeric>
#include <TH1F.h>
#include <TH2F.h>
#include <THStack.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TLine.h>
#include <TStyle.h>
#include <TNtuple.h>

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;
using PVertex = o2::dataformats::PrimaryVertex;
using V0 = o2::dataformats::V0;
using ITSCluster = o2::BaseCluster<float>;
using mask_t = o2::dataformats::GlobalTrackID::mask_t;
using Track = o2::track::TrackParCov;
using TrackITS = o2::its::TrackITS;
using TrackTPC = o2::tpc::TrackTPC;
using TrackITSTPC = o2::dataformats::TrackTPCITS;
using PIDResponse = o2::tpc::PIDResponse;
using DCA = o2::dataformats::DCA;
using PID = o2::track::PID;

// structure for storing the output tree
struct particle {
  // mc properties
  int pdg = -1;
  bool fakeMatch = 0;
  // reco properties
  int sign = -1;
  float p, pt, pTPC, pITS, eta, phi, tgL, chi2ITS, chi2TPC, chi2ITSTPC;
  int nClusTPC;
  float dEdx, nSigmaDeu, nSigmaP, nSigmaK, nSigmaPi, nSigmaE;
  int clSizeL0, clSizeL1, clSizeL2, clSizeL3, clSizeL4, clSizeL5, clSizeL6;
  std::array<int, 7> clSizesITS;
};

class PIDStudy : public Task
{
 public:
  PIDStudy(std::shared_ptr<DataRequest> dr,
           std::shared_ptr<o2::base::GRPGeomRequest> gr,
           bool isMC,
           std::shared_ptr<o2::steer::MCKinematicsReader> kineReader) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC), mKineReader(kineReader){};
  ~PIDStudy() final = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

 private:
  // Other functions
  void process(o2::globaltracking::RecoContainer&);
  void loadData(o2::globaltracking::RecoContainer&);

  // Helper functions
  void saveOutput();
  void updateTimeDependentParams(ProcessingContext& pc);
  void getClusterSizes(std::vector<int>&, const gsl::span<const o2::itsmft::CompClusterExt>, gsl::span<const unsigned char>::iterator&, const o2::itsmft::TopologyDictionary*);
  std::array<int, 7> getTrackClusterSizes(const TrackITS& track);
  float computeNSigma(PID pid, TrackTPC& tpcTrack, float resolution);

  // Running options
  bool mUseMC;

  PIDResponse mPIDresponse;
  float mBBres;
  // Data
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<int> mClusterSizes;
  gsl::span<const o2::itsmft::CompClusterExt> mClusters;
  gsl::span<const int> mInputITSidxs;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::string mOutName;
  std::shared_ptr<o2::steer::MCKinematicsReader> mKineReader;
};

void PIDStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  LOGP(info, "Starting average cluster size study...");

  if (mUseMC) { // for counting the missed K0shorts
    mKineReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
  }
  auto& params = o2::its::study::PIDStudyParamConfig::Instance();
  mOutName = params.outFileName;
  mPIDresponse.setBetheBlochParams(params.mBBpars);
  mBBres = params.mBBres;
  LOGP(info, "PID size study initialized.");

  // prepare output tree
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.c_str(), "recreate");
}

void PIDStudy::run(ProcessingContext& pc)
{
  // auto geom = o2::its::GeometryTGeo::Instance();
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void PIDStudy::getClusterSizes(std::vector<int>& clusSizeVec, const gsl::span<const o2::itsmft::CompClusterExt> ITSclus, gsl::span<const unsigned char>::iterator& pattIt, const o2::itsmft::TopologyDictionary* mdict)
{
  for (unsigned int iClus{0}; iClus < ITSclus.size(); ++iClus) {
    auto& clus = ITSclus[iClus];
    auto pattID = clus.getPatternID();
    int npix;
    o2::itsmft::ClusterPattern patt;

    if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict->isGroup(pattID)) {
      patt.acquirePattern(pattIt);
      npix = patt.getNPixels();
    } else {
      npix = mdict->getNpixels(pattID);
      patt = mdict->getPattern(pattID);
    }
    clusSizeVec[iClus] = npix;
  }
}

void PIDStudy::loadData(o2::globaltracking::RecoContainer& recoData)
{
  mInputITSidxs = recoData.getITSTracksClusterRefs();
  mClusters = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();
  mClusterSizes.resize(mClusters.size());
  auto pattIt = clusPatt.begin();
  getClusterSizes(mClusterSizes, mClusters, pattIt, mDict);
}

void PIDStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  loadData(recoData);
  auto ITSTPCtracks = recoData.getTPCITSTracks();
  LOGP(debug, "Found {} ITSTPC tracks.", ITSTPCtracks.size());

  gsl::span<const o2::MCCompLabel> mcLabelsITS, mcLabelsTPC;
  if (mUseMC) {
    mcLabelsITS = recoData.getITSTracksMCLabels();
    mcLabelsTPC = recoData.getTPCTracksMCLabels();
    LOGP(debug, "Found {} ITS labels.", mcLabelsITS.size());
    LOGP(debug, "Found {} TPC labels.", mcLabelsTPC.size());
  }

  for (unsigned int iTrack{0}; iTrack < ITSTPCtracks.size(); ++iTrack) {

    auto& ITSTPCtrack = ITSTPCtracks[iTrack];
    if (ITSTPCtrack.getRefITS().getSource() == GTrackID::ITSAB) { // excluding Afterburned tracks
      continue;
    }
    particle part;
    auto ITStrack = recoData.getITSTrack(ITSTPCtrack.getRefITS());
    auto TPCtrack = recoData.getTPCTrack(ITSTPCtrack.getRefTPC());

    if (mUseMC) {
      // MC info
      auto& mcLabelITS = mcLabelsITS[ITSTPCtrack.getRefITS().getIndex()];
      auto& mcLabelTPC = mcLabelsTPC[ITSTPCtrack.getRefTPC().getIndex()];
      if (mcLabelITS.getTrackID() != (int)mcLabelTPC.getTrackID()) {
        part.fakeMatch = 1;
      }
      auto mctrk = mKineReader->getTrack(mcLabelITS);
      part.pdg = mctrk->GetPdgCode();
    }

    part.sign = ITSTPCtrack.getSign();
    part.clSizesITS = getTrackClusterSizes(ITStrack);
    part.p = ITSTPCtrack.getP();
    part.pt = ITSTPCtrack.getPt();
    part.pTPC = TPCtrack.getP();
    part.pITS = ITStrack.getP();
    part.eta = ITSTPCtrack.getEta();
    part.phi = ITSTPCtrack.getPhi();
    part.tgL = ITSTPCtrack.getTgl();
    part.chi2ITS = ITStrack.getChi2();
    part.chi2TPC = TPCtrack.getChi2();
    part.chi2ITSTPC = ITSTPCtrack.getChi2Match();

    part.clSizeL0 = part.clSizesITS[0];
    part.clSizeL1 = part.clSizesITS[1];
    part.clSizeL2 = part.clSizesITS[2];
    part.clSizeL3 = part.clSizesITS[3];
    part.clSizeL4 = part.clSizesITS[4];
    part.clSizeL5 = part.clSizesITS[5];
    part.clSizeL6 = part.clSizesITS[6];

    // PID info
    part.dEdx = TPCtrack.getdEdx().dEdxTotTPC;
    part.nClusTPC = TPCtrack.getNClusters();
    // 7% resolution for all particles
    part.nSigmaDeu = computeNSigma(PID::Deuteron, TPCtrack, mBBres);
    part.nSigmaP = computeNSigma(PID::Proton, TPCtrack, mBBres);
    part.nSigmaK = computeNSigma(PID::Kaon, TPCtrack, mBBres);
    part.nSigmaPi = computeNSigma(PID::Pion, TPCtrack, mBBres);
    part.nSigmaE = computeNSigma(PID::Electron, TPCtrack, mBBres);

    if (mUseMC) {
      (*mDBGOut) << "outTree"
                 << "pdg=" << part.pdg << "fakeMatch=" << part.fakeMatch;
    }
    (*mDBGOut) << "outTree"
               << "sign=" << part.sign << "p=" << part.p << "pt=" << part.pt << "pTPC=" << part.pTPC << "pITS=" << part.pITS
               << "eta=" << part.eta << "phi=" << part.phi << "tgL=" << part.tgL << "chi2ITS=" << part.chi2ITS << "chi2TPC="
               << part.chi2TPC << "chi2ITSTPC=" << part.chi2ITSTPC << "dEdx=" << part.dEdx << "nClusTPC=" << part.nClusTPC
               << "nSigmaDeu=" << part.nSigmaDeu << "nSigmaP=" << part.nSigmaP << "nSigmaK=" << part.nSigmaK << "nSigmaPi="
               << part.nSigmaPi << "nSigmaE=" << part.nSigmaE << "clSizeL0=" << part.clSizeL0 << "clSizeL1=" << part.clSizeL1
               << "clSizeL2=" << part.clSizeL2 << "clSizeL3=" << part.clSizeL3 << "clSizeL4=" << part.clSizeL4 << "clSizeL5="
               << part.clSizeL5 << "clSizeL6=" << part.clSizeL6 << "\n";
  }
}

std::array<int, 7> PIDStudy::getTrackClusterSizes(const TrackITS& track)
{
  auto geom = o2::its::GeometryTGeo::Instance();
  std::array<int, 7> clusSizes = {-1, -1, -1, -1, -1, -1, -1};
  auto firstClus = track.getFirstClusterEntry();
  auto ncl = track.getNumberOfClusters();
  for (int icl = 0; icl < ncl; icl++) {
    auto& clus = mClusters[mInputITSidxs[firstClus + icl]];
    auto& clSize = mClusterSizes[mInputITSidxs[firstClus + icl]];
    auto layer = geom->getLayer(clus.getSensorID());
    clusSizes[layer] = clSize;
  }
  return clusSizes;
}

void PIDStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this param need to be queried only once
    initOnceDone = true;
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void PIDStudy::saveOutput()
{
  mDBGOut.reset();
  LOGP(info, "Stored histograms into {}", mOutName.c_str());
}

void PIDStudy::endOfStream(EndOfStreamContext& ec)
{
  // saveOutput();
}

void PIDStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  // o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

float PIDStudy::computeNSigma(PID pid, TrackTPC& tpcTrack, float resolution)
{
  float nSigma = -999;
  float bb = mPIDresponse.getExpectedSignal(tpcTrack, pid);
  if (tpcTrack.getdEdx().dEdxTotTPC > 0) {
    nSigma = (tpcTrack.getdEdx().dEdxTotTPC - bb) / (resolution * bb);
  }
  return nSigma;
}

DataProcessorSpec getPIDStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestClusters(srcClustersMask, useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "its-pid-study",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PIDStudy>(dataRequest, ggRequest, useMC, kineReader)},
    Options{}};
}
} // namespace study
} // namespace its
} // namespace o2
