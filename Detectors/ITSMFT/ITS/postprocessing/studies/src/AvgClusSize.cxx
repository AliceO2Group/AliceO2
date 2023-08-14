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

/// \file AvgClusSize.cxx
/// \brief Study to calculate average cluster size per track in the ITS
/// \author Tucker Hwang mhwang@cern.ch

#include "ITSStudies/AvgClusSize.h"
#include "ITSStudies/ITSStudiesConfigParam.h"

#include "Framework/Task.h"
#include "ITSBase/GeometryTGeo.h"
#include "Steer/MCKinematicsReader.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITStracking/IOUtils.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITS/TrackITS.h"
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
using DCA = o2::dataformats::DCA;
using PID = o2::track::PID;

class AvgClusSizeStudy : public Task
{
 public:
  AvgClusSizeStudy(std::shared_ptr<DataRequest> dr,
                   std::shared_ptr<o2::base::GRPGeomRequest> gr,
                   bool isMC,
                   std::shared_ptr<o2::steer::MCKinematicsReader> kineReader) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC(isMC), mKineReader(kineReader){};
  ~AvgClusSizeStudy() final = default;
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
  void prepareOutput();
  void setStyle();
  void updateTimeDependentParams(ProcessingContext& pc);
  float getAverageClusterSize(o2::its::TrackITS*);
  float calcV0HypoMass(const V0&, PID, PID);
  void calcAPVars(const V0&, float*, float*);
  void getClusterSizes(std::vector<int>&, const gsl::span<const o2::itsmft::CompClusterExt>, gsl::span<const unsigned char>::iterator&, const o2::itsmft::TopologyDictionary*);
  void saveHistograms();
  void plotHistograms();
  void fillEtaBin(float eta, float clusSize, int i);

  // Running options
  bool mUseMC;

  // Data
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<int> mClusterSizes;
  gsl::span<const int> mInputITSidxs;
  std::vector<o2::MCTrack> mMCTracks;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  // Output plots
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::unique_ptr<TNtuple> mOutputNtupleAll;
  std::unique_ptr<TNtuple> mOutputNtupleCut;

  std::unique_ptr<THStack> mMCR{};
  std::unique_ptr<THStack> mMCCosPA{};
  std::unique_ptr<THStack> mMCPosACS{};
  std::unique_ptr<THStack> mMCNegACS{};
  std::unique_ptr<THStack> mMCDauDCA{};
  std::unique_ptr<THStack> mMCPosPVDCA{};
  std::unique_ptr<THStack> mMCNegPVDCA{};
  std::unique_ptr<THStack> mMCV0PVDCA{};

  std::unique_ptr<TH1F> mMCRisTg{};
  std::unique_ptr<TH1F> mMCRisBg{};
  std::unique_ptr<TH1F> mMCCosPAisTg{};
  std::unique_ptr<TH1F> mMCCosPAisBg{};
  std::unique_ptr<TH1F> mMCPosACSisTg{};
  std::unique_ptr<TH1F> mMCPosACSisBg{};
  std::unique_ptr<TH1F> mMCNegACSisTg{};
  std::unique_ptr<TH1F> mMCNegACSisBg{};
  std::unique_ptr<TH1F> mMCDauDCAisTg{};
  std::unique_ptr<TH1F> mMCDauDCAisBg{};
  std::unique_ptr<TH1F> mMCPosPVDCAisTg{};
  std::unique_ptr<TH1F> mMCPosPVDCAisBg{};
  std::unique_ptr<TH1F> mMCNegPVDCAisTg{};
  std::unique_ptr<TH1F> mMCNegPVDCAisBg{};
  std::unique_ptr<TH1F> mMCV0PVDCAisTg{};
  std::unique_ptr<TH1F> mMCV0PVDCAisBg{};

  std::unique_ptr<TH2F> mMCArmPodolisTg{};
  std::unique_ptr<TH2F> mMCArmPodolisBg{};

  std::unique_ptr<THStack> mAvgClusSizeCEta{};
  std::vector<std::unique_ptr<TH1F>> mAvgClusSizeCEtaVec{};

  std::vector<float> mEtaBinUL; // upper edges for eta bins

  // Counters for target V0 identification
  int nNotValid = 0;
  int nNullptrs = 0;
  int nMotherIDMismatch = 0;
  int nEvIDMismatch = 0;
  int nTargetV0 = 0;
  int nNotTargetV0 = 0;
  int nV0OutOfEtaRange = 0;

  std::string mOutName;
  std::shared_ptr<o2::steer::MCKinematicsReader> mKineReader;
};

void AvgClusSizeStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  LOGP(info, "Starting average cluster size study...");

  prepareOutput();

  if (mUseMC) { // for counting the missed K0shorts
    mKineReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
    for (int iEvent{0}; iEvent < mKineReader->getNEvents(0); iEvent++) {
      auto mctrk = mKineReader->getTracks(0, iEvent);
      mMCTracks.insert(mMCTracks.end(), mctrk.begin(), mctrk.end());
    }
  }

  LOGP(info, "Cluster size study initialized.");
}

void AvgClusSizeStudy::prepareOutput()
{
  auto& params = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
  mOutName = params.outFileName;
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.c_str(), "recreate");
  mOutputNtupleAll = std::make_unique<TNtuple>("v0_data", "v0 data", "dPosACS:dNegACS:cosPA:V0R:eta:dauDCA:dPospvDCA:dNegpvDCA:v0pvDCA:alpha:pT:K0mass:lambdaMass:antilambdaMass:v0PDGcode");
  mOutputNtupleCut = std::make_unique<TNtuple>("cut_v0_data", "v0 data (cut)", "dPosACS:dNegACS:cosPA:V0R:eta:dauDCA:dPospvDCA:dNegpvDCA:v0pvDCA:alpha:pT:K0mass:lambdaMass:antilambdaMass:v0PDGcode");

  mMCR = std::make_unique<THStack>("R", "V0 decay length R;R (cm?)");
  mMCCosPA = std::make_unique<THStack>("cosPA", "cos(#theta_{p})");
  mMCPosACS = std::make_unique<THStack>("acsPos", "Average cluster size per track;pixels / cluster / track");
  mMCNegACS = std::make_unique<THStack>("acsNeg", "Average cluster size per track;pixels / cluster / track");
  mMCDauDCA = std::make_unique<THStack>("dauDCA", "Prong-prong DCA;cm?");
  mMCPosPVDCA = std::make_unique<THStack>("posPVDCA", "Positive prong-primary vertex DCA;cm?");
  mMCNegPVDCA = std::make_unique<THStack>("negPVDCA", "Negative prong-primary vertex DCA;cm?");
  mMCV0PVDCA = std::make_unique<THStack>("v0PVDCA", "V0 reconstructed track-primary vertex DCA;cm?");

  mMCRisTg = std::make_unique<TH1F>("mcRTg", "target V0", 40, 0, 2);
  mMCRisBg = std::make_unique<TH1F>("mcRBg", "background", 100, 0, 2);
  mMCCosPAisTg = std::make_unique<TH1F>("mcCosPATg", "target V0", 40, 0, 1);
  mMCCosPAisBg = std::make_unique<TH1F>("mcCosPABg", "background", 100, 0, 1);
  mMCPosACSisTg = std::make_unique<TH1F>("mcPosACSTg", "target V0", 60, 0, 30);
  mMCPosACSisBg = std::make_unique<TH1F>("mcPosACSBg", "background", 150, 0, 30);
  mMCNegACSisTg = std::make_unique<TH1F>("mcNegACSTg", "target V0", 60, 0, 30);
  mMCNegACSisBg = std::make_unique<TH1F>("mcNegACSBg", "background", 150, 0, 30);
  mMCDauDCAisTg = std::make_unique<TH1F>("mcDauDCATg", "target V0", 40, 0, 2);
  mMCDauDCAisBg = std::make_unique<TH1F>("mcDauDCABg", "background", 100, 0, 2);
  mMCPosPVDCAisTg = std::make_unique<TH1F>("mcPosPVDCATg", "target V0", 40, 0, 15);
  mMCPosPVDCAisBg = std::make_unique<TH1F>("mcPosPVDCABg", "background", 100, 0, 15);
  mMCNegPVDCAisTg = std::make_unique<TH1F>("mcNegPVDCATg", "target V0", 40, 0, 15);
  mMCNegPVDCAisBg = std::make_unique<TH1F>("mcNegPVDCABg", "background", 100, 0, 15);
  mMCV0PVDCAisTg = std::make_unique<TH1F>("mcV0PVDCATg", "target V0", 40, 0, 15);
  mMCV0PVDCAisBg = std::make_unique<TH1F>("mcV0PVDCABg", "background", 100, 0, 15);

  mMCArmPodolisTg = std::make_unique<TH2F>("apPlotTg", "Armenteros-Podolanski;#alpha_{Arm};p_{T}^{Arm} (GeV/c)", 150, -2, 2, 150, 0, 0.5);
  mMCArmPodolisBg = std::make_unique<TH2F>("apPlotBg", "Armenteros-Podolanski;#alpha_{Arm};p_{T}^{Arm} (GeV/c)", 150, -2, 2, 150, 0, 0.5);

  mAvgClusSizeCEta = std::make_unique<THStack>("avgclussizeeta", "Average cluster size per track;pixels / cluster / track"); // auto-set axis ranges
  float binWidth = (params.etaMax - params.etaMin) / (float)params.etaNBins;
  mEtaBinUL.reserve(params.etaNBins);
  for (int i = 0; i < params.etaNBins; i++) {
    mEtaBinUL.emplace_back(params.etaMin + (binWidth * (i + 1)));
    mAvgClusSizeCEtaVec.push_back(std::make_unique<TH1F>(Form("avgclussize%i", i), Form("%.2f < #eta < %.2f", mEtaBinUL[i] - binWidth, mEtaBinUL[i]), params.sizeNBins, 0, params.sizeMax));
    mAvgClusSizeCEtaVec[i]->SetDirectory(nullptr);
    mAvgClusSizeCEta->Add(mAvgClusSizeCEtaVec[i].get());
  }

  mMCRisTg->SetDirectory(nullptr);
  mMCRisBg->SetDirectory(nullptr);
  mMCCosPAisTg->SetDirectory(nullptr);
  mMCCosPAisBg->SetDirectory(nullptr);
  mMCDauDCAisTg->SetDirectory(nullptr);
  mMCDauDCAisBg->SetDirectory(nullptr);
  mMCPosACSisTg->SetDirectory(nullptr);
  mMCPosACSisBg->SetDirectory(nullptr);
  mMCNegACSisTg->SetDirectory(nullptr);
  mMCNegACSisBg->SetDirectory(nullptr);
  mMCPosPVDCAisTg->SetDirectory(nullptr);
  mMCPosPVDCAisBg->SetDirectory(nullptr);
  mMCNegPVDCAisTg->SetDirectory(nullptr);
  mMCNegPVDCAisBg->SetDirectory(nullptr);
  mMCV0PVDCAisTg->SetDirectory(nullptr);
  mMCV0PVDCAisBg->SetDirectory(nullptr);

  mMCArmPodolisTg->SetDirectory(nullptr);
  mMCArmPodolisBg->SetDirectory(nullptr);

  mOutputNtupleAll->SetDirectory(nullptr);
  mOutputNtupleCut->SetDirectory(nullptr);

  mMCR->Add(mMCRisTg.get());
  mMCR->Add(mMCRisBg.get());
  mMCCosPA->Add(mMCCosPAisTg.get());
  mMCCosPA->Add(mMCCosPAisBg.get());
  mMCPosACS->Add(mMCPosACSisTg.get());
  mMCPosACS->Add(mMCPosACSisBg.get());
  mMCNegACS->Add(mMCNegACSisTg.get());
  mMCNegACS->Add(mMCNegACSisBg.get());
  mMCDauDCA->Add(mMCDauDCAisTg.get());
  mMCDauDCA->Add(mMCDauDCAisBg.get());
  mMCPosPVDCA->Add(mMCPosPVDCAisTg.get());
  mMCPosPVDCA->Add(mMCPosPVDCAisBg.get());
  mMCNegPVDCA->Add(mMCNegPVDCAisTg.get());
  mMCNegPVDCA->Add(mMCNegPVDCAisBg.get());
  mMCV0PVDCA->Add(mMCV0PVDCAisTg.get());
  mMCV0PVDCA->Add(mMCV0PVDCAisBg.get());
}

void AvgClusSizeStudy::setStyle()
{
  gStyle->SetPalette(kRainbow);
  std::vector<int> colors = {1, 2, 3, 4, 6, 7, 41, 47};
  std::vector<int> markers = {2, 3, 4, 5, 25, 26, 27, 28, 32};
  for (int i = 0; i < mAvgClusSizeCEtaVec.size(); i++) {
    mAvgClusSizeCEtaVec[i]->SetMarkerStyle(markers[i]);
    mAvgClusSizeCEtaVec[i]->SetMarkerColor(colors[i]);
    mAvgClusSizeCEtaVec[i]->SetLineColor(colors[i]);
  }

  mMCRisTg->SetLineColor(kRed);
  mMCCosPAisTg->SetLineColor(kRed);
  mMCPosACSisTg->SetLineColor(kRed);
  mMCNegACSisTg->SetLineColor(kRed);
  mMCDauDCAisTg->SetLineColor(kRed);
  mMCPosPVDCAisTg->SetLineColor(kRed);
  mMCNegPVDCAisTg->SetLineColor(kRed);
  mMCV0PVDCAisTg->SetLineColor(kRed);
}

void AvgClusSizeStudy::run(ProcessingContext& pc)
{
  // auto geom = o2::its::GeometryTGeo::Instance();
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  process(recoData);
}

void AvgClusSizeStudy::getClusterSizes(std::vector<int>& clusSizeVec, const gsl::span<const o2::itsmft::CompClusterExt> ITSclus, gsl::span<const unsigned char>::iterator& pattIt, const o2::itsmft::TopologyDictionary* mdict)
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

void AvgClusSizeStudy::loadData(o2::globaltracking::RecoContainer& recoData)
{
  mInputITSidxs = recoData.getITSTracksClusterRefs();
  auto compClus = recoData.getITSClusters();
  auto clusPatt = recoData.getITSClustersPatterns();
  mClusterSizes.resize(compClus.size());
  auto pattIt = clusPatt.begin();
  getClusterSizes(mClusterSizes, compClus, pattIt, mDict);
}

void AvgClusSizeStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  auto& params = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
  float dPosACS, dNegACS, dauDCA, cosPA, v0R, eta, dPospvDCA, dNegpvDCA, v0pvDCA, tgV0HypoMass, bgV0HypoMass, alphaArm, pT; // ACS=average cluster size
  bool isMCTarget = false;
  int targetPDGCode = 310;
  TrackITS dPosRecoTrk, dNegRecoTrk; // daughter ITS tracks
  DCA dPosDCA, dNegDCA, v0DCA;       // DCA object for prong to primary vertex (DCA = o2::dataformats::DCA)
  PVertex pv;

  PID targetV0 = PID("K0");
  PID backgroundV0 = PID("Lambda");
  PID tgPos, tgNeg, bgPos, bgNeg;
  if (params.targetV0 == "K0") {
    bgPos = PID("Proton");
    LOGP(info, "V0 target set to K0-short.");
  } else if (params.targetV0 == "Lambda") {
    targetV0 = PID("Lambda");
    tgPos = PID("Proton");
    backgroundV0 = PID("K0");
    targetPDGCode = 3122;
    LOGP(info, "V0 target set to Lambda.");
  } else {
    LOGP(warning, "Given V0 target not recognized, defaulting to K0-short.");
    bgPos = PID("Proton");
  }

  // Variables for MC analysis
  gsl::span<const o2::MCCompLabel> mcLabels;
  o2::MCCompLabel dPosLab, dNegLab;
  const o2::MCTrack *V0mcTrk, *dPosMCTrk, *dNegMCTrk;
  int V0PdgCode, mPosTrkId, mNegTrkId;

  loadData(recoData);
  auto V0s = recoData.getV0s();
  auto V0sIdx = recoData.getV0sIdx();
  size_t nV0s = V0sIdx.size();
  if (nV0s && nV0s != V0s.size()) {
    LOGP(fatal, "This data has not secondary vertices kinematics filled");
  }

  LOGP(info, "Found {} reconstructed V0s.", nV0s);
  LOGP(info, "Found {} ITS tracks.", recoData.getITSTracks().size());
  LOGP(info, "Found {} ROFs.", recoData.getITSTracksROFRecords().size());
  if (mUseMC) {
    mcLabels = recoData.getITSTracksMCLabels();
    LOGP(info, "Found {} labels.", mcLabels.size());
  }

  for (size_t iv = 0; iv < nV0s; iv++) {
    auto v0 = V0s[iv];
    const auto& v0Idx = V0sIdx[iv];
    dPosRecoTrk = recoData.getITSTrack(v0Idx.getProngID(0)); // cannot use v0.getProng() since it returns TrackParCov not TrackITS
    dNegRecoTrk = recoData.getITSTrack(v0Idx.getProngID(1));

    pv = recoData.getPrimaryVertex(v0Idx.getVertexID()); // extract primary vertex
    dPosRecoTrk.propagateToDCA(pv, params.b, &dPosDCA); // calculate and store DCA objects for both prongs
    dNegRecoTrk.propagateToDCA(pv, params.b, &dNegDCA);
    v0.propagateToDCA(pv, params.b, &v0DCA);

    dPospvDCA = std::sqrt(dPosDCA.getR2()); // extract DCA distance from DCA object
    dNegpvDCA = std::sqrt(dNegDCA.getR2());
    v0pvDCA = std::sqrt(v0DCA.getR2());

    eta = v0.getEta();
    dauDCA = v0.getDCA();
    cosPA = v0.getCosPA();
    v0R = std::sqrt(v0.calcR2()); // gives distance from pvertex to origin? in centimeters (?) NOTE: unsure if this is to the primary vertex or to origin

    dPosACS = getAverageClusterSize(&dPosRecoTrk);
    dNegACS = getAverageClusterSize(&dNegRecoTrk);

    tgV0HypoMass = calcV0HypoMass(v0, tgPos, tgNeg);
    bgV0HypoMass = calcV0HypoMass(v0, bgPos, bgNeg);
    calcAPVars(v0, &alphaArm, &pT);

    if (mUseMC) { // check whether queried V0 is the target V0 in MC, and fill the cut validation plots
      V0PdgCode = 0;
      isMCTarget = false;
      dPosLab = mcLabels[v0Idx.getProngID(0)]; // extract MC labels for the prongs
      dNegLab = mcLabels[v0Idx.getProngID(1)];
      if (!dPosLab.isValid() || !dNegLab.isValid()) {
        LOGP(debug, "Daughter MCCompLabel not valid: {}(+) and {}(-). Skipping.", dPosLab.isValid(), dNegLab.isValid());
        nNotValid++;
      } else {
        dPosMCTrk = mKineReader->getTrack(dPosLab);
        dNegMCTrk = mKineReader->getTrack(dNegLab);
        if (dPosMCTrk == nullptr || dNegMCTrk == nullptr) {
          LOGP(debug, "Nullptr found: {}(+) and {}(-). Skipping.", (void*)dPosMCTrk, (void*)dNegMCTrk);
          nNullptrs++;
        } else {
          mPosTrkId = dPosMCTrk->getMotherTrackId();
          mNegTrkId = dNegMCTrk->getMotherTrackId();
          LOGP(debug, "Daughter PDG codes: {}(+) and {}(-)", dPosMCTrk->GetPdgCode(), dNegMCTrk->GetPdgCode());
          if (mPosTrkId != mNegTrkId || mPosTrkId == -1 || mNegTrkId == -1) {
            LOGP(debug, "Mother track ID mismatch or default -1: {}(+) and {}(-). Skipping.", mPosTrkId, mNegTrkId);
            nMotherIDMismatch++;
          } else {
            if (dNegLab.getEventID() != dPosLab.getEventID()) {
              LOGP(debug, "Daughter EvID mismatch: {}(+) and {}(-). Skipping.", dPosLab.getEventID(), dNegLab.getEventID());
              nEvIDMismatch++;
            } else {
              V0mcTrk = mKineReader->getTrack(dNegLab.getEventID(), mPosTrkId); // assume daughter MCTracks are in same event as V0 MCTrack
              V0PdgCode = V0mcTrk->GetPdgCode();
              if (V0PdgCode == targetPDGCode) {
                isMCTarget = true;
                nTargetV0++;
              } else {
                nNotTargetV0++;
              }
            }
          }
        }
      }
      if (isMCTarget) {
        mMCCosPAisTg->Fill(cosPA);
        mMCDauDCAisTg->Fill(dauDCA);
        mMCRisTg->Fill(v0R);
        mMCPosPVDCAisTg->Fill(dPospvDCA);
        mMCNegPVDCAisTg->Fill(dNegpvDCA);
        mMCPosACSisTg->Fill(dPosACS);
        mMCNegACSisTg->Fill(dNegACS);
        mMCV0PVDCAisTg->Fill(v0pvDCA);
        mMCArmPodolisTg->Fill(alphaArm, pT);
      } else {
        mMCCosPAisBg->Fill(cosPA);
        mMCDauDCAisBg->Fill(dauDCA);
        mMCRisBg->Fill(v0R);
        mMCPosPVDCAisBg->Fill(dPospvDCA);
        mMCNegPVDCAisBg->Fill(dNegpvDCA);
        mMCPosACSisBg->Fill(dPosACS);
        mMCNegACSisBg->Fill(dNegACS);
        mMCV0PVDCAisBg->Fill(v0pvDCA);
        mMCArmPodolisBg->Fill(alphaArm, pT);
      }
    }

    mOutputNtupleAll->Fill(dPosACS, dNegACS, cosPA, v0R, eta, dauDCA, dPospvDCA, dNegpvDCA, v0pvDCA, alphaArm, pT, calcV0HypoMass(v0, PID::Pion, PID::Pion), calcV0HypoMass(v0, PID::Proton, PID::Pion), calcV0HypoMass(v0, PID::Pion, PID::Proton), (float)V0PdgCode);
    if ((cosPA > params.cosPAmin || params.disableCosPA) && (v0R < params.Rmax || params.disableRmax) && (v0R > params.Rmin || params.disableRmin) && (dauDCA < params.prongDCAmax || params.disableProngDCAmax) && (dPospvDCA > params.dauPVDCAmin || params.disableDauPVDCAmin) && (dNegpvDCA > params.dauPVDCAmin || params.disableDauPVDCAmin) && (v0pvDCA < params.v0PVDCAmax || params.disableV0PVDCAmax) && (abs(bgV0HypoMass - backgroundV0.getMass()) > params.bgV0window || params.disableMassHypoth) && (abs(tgV0HypoMass - targetV0.getMass()) < params.tgV0window || params.disableMassHypoth)) {
      mOutputNtupleCut->Fill(dPosACS, dNegACS, cosPA, v0R, eta, dauDCA, dPospvDCA, dNegpvDCA, v0pvDCA, alphaArm, pT, calcV0HypoMass(v0, PID::Pion, PID::Pion), calcV0HypoMass(v0, PID::Proton, PID::Pion), calcV0HypoMass(v0, PID::Pion, PID::Proton), (float)V0PdgCode);
      if (eta > params.etaMin && eta < params.etaMax) {
        fillEtaBin(eta, dPosACS, 0);
        fillEtaBin(eta, dNegACS, 0);
      } else {
        nV0OutOfEtaRange++;
      }
    }
  }

  if (mUseMC) {
    LOGP(info, "MONTE CARLO OVERALL STATISTICS: {} total V0s, {} nonvalid daughter labels, {} nullptrs, {} motherID mismatches, {} evID mismatches, {} matching target, {} not matching target", V0s.size(), nNotValid, nNullptrs, nMotherIDMismatch, nEvIDMismatch, nTargetV0, nNotTargetV0);
    int nPrimaryTargetV0 = 0;
    for (auto& mcTrk : mMCTracks) { // search through all MC tracks to find primary target V0s, whether reconstructed or not
      if (mcTrk.GetPdgCode() == targetPDGCode && mcTrk.isPrimary()) {
        nPrimaryTargetV0++;
      }
    }
    LOGP(info, "MONTE CARLO OVERALL STATISTICS: {} MC target V0s (isPrimary) found in MC tracks out of {} total MC tracks", nPrimaryTargetV0, mMCTracks.size());
  }
  LOGP(info, "{} V0s out of eta range ({}, {})", nV0OutOfEtaRange, params.etaMin, params.etaMax);
}

float AvgClusSizeStudy::getAverageClusterSize(TrackITS* daughter)
{
  int totalSize{0};
  auto firstClus = daughter->getFirstClusterEntry();
  auto ncl = daughter->getNumberOfClusters();
  for (int icl = 0; icl < ncl; icl++) {
    totalSize += mClusterSizes[mInputITSidxs[firstClus + icl]];
  }
  return (float)totalSize / (float)ncl;
}

float AvgClusSizeStudy::calcV0HypoMass(const V0& v0, PID hypothPIDPos, PID hypothPIDNeg)
{
  // Mass hypothesis calculation; taken from o2::strangeness_tracking::StrangenessTracker::calcMotherMass()
  std::array<float, 3> pPos, pNeg, pV0;
  v0.getProng(0).getPxPyPzGlo(pPos);
  v0.getProng(1).getPxPyPzGlo(pNeg);
  v0.getPxPyPzGlo(pV0);
  double m2Pos = PID::getMass2(hypothPIDPos);
  double m2Neg = PID::getMass2(hypothPIDNeg);
  double p2Pos = (pPos[0] * pPos[0]) + (pPos[1] * pPos[1]) + (pPos[2] * pPos[2]);
  double p2Neg = (pNeg[0] * pNeg[0]) + (pNeg[1] * pNeg[1]) + (pNeg[2] * pNeg[2]);
  double ePos = std::sqrt(p2Pos + m2Pos), eNeg = std::sqrt(p2Neg + m2Neg);
  double e2V0 = (ePos + eNeg) * (ePos + eNeg);
  double pxV0 = (pPos[0] + pNeg[0]);
  double pyV0 = (pPos[1] + pNeg[1]);
  double pzV0 = (pPos[2] + pNeg[2]);
  double p2V0 = (pxV0 * pxV0) + (pyV0 * pyV0) + (pzV0 * pzV0);
  return (float)std::sqrt(e2V0 - p2V0);
}

void AvgClusSizeStudy::calcAPVars(const V0& v0, float* alphaArm, float* pT)
{
  // Calculation of the Armenteros-Podolanski variables
  std::array<float, 3> pV0, pPos, pNeg;
  v0.getProng(0).getPxPyPzGlo(pPos);
  v0.getProng(1).getPxPyPzGlo(pNeg);
  v0.getPxPyPzGlo(pV0);
  double p2V0 = pV0[0] * pV0[0] + pV0[1] * pV0[1] + pV0[2] * pV0[2];
  double qNeg = pNeg[0] * pV0[0] + pNeg[1] * pV0[1] + pNeg[2] * pV0[2];
  double qPos = pPos[0] * pV0[0] + pPos[1] * pV0[1] + pPos[2] * pV0[2];
  *alphaArm = (float)(qPos - qNeg) / (qPos + qNeg);
  double p2Pos = pPos[0] * pPos[0] + pPos[1] * pPos[1] + pPos[2] * pPos[2];
  *pT = (float)std::sqrt(p2Pos - ((qPos * qPos) / p2V0));
};

void AvgClusSizeStudy::fillEtaBin(float eta, float clusSize, int i)
{
  if (eta < mEtaBinUL[i]) {
    mAvgClusSizeCEtaVec[i]->Fill(clusSize);
  } else {
    fillEtaBin(eta, clusSize, i + 1);
  }
}

void AvgClusSizeStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this param need to be queried only once
    initOnceDone = true;
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void AvgClusSizeStudy::saveHistograms()
{
  mDBGOut.reset();
  TFile fout(mOutName.c_str(), "RECREATE");

  fout.WriteTObject(mOutputNtupleAll.get());
  fout.WriteTObject(mOutputNtupleCut.get());
  fout.WriteTObject(mMCR.get());
  fout.WriteTObject(mMCCosPA.get());
  fout.WriteTObject(mMCPosACS.get());
  fout.WriteTObject(mMCNegACS.get());
  fout.WriteTObject(mMCDauDCA.get());
  fout.WriteTObject(mMCPosPVDCA.get());
  fout.WriteTObject(mMCNegPVDCA.get());
  fout.WriteTObject(mMCV0PVDCA.get());
  fout.WriteTObject(mMCArmPodolisTg.get());
  fout.WriteTObject(mMCArmPodolisBg.get());

  fout.WriteTObject(mAvgClusSizeCEta.get());
  fout.Close();

  LOGP(info, "Stored histograms into {}", mOutName.c_str());
}

void AvgClusSizeStudy::plotHistograms()
{
  if (mUseMC) {
    TCanvas* cMCR = new TCanvas();
    mMCR->Draw("nostack");
    cMCR->BuildLegend();
    cMCR->Print("mcR.png");
    TCanvas* cMCCosPA = new TCanvas();
    mMCCosPA->Draw("nostack");
    cMCCosPA->BuildLegend();
    cMCCosPA->Print("mcCosPA.png");
    TCanvas* cMCPosACS = new TCanvas();
    mMCPosACS->Draw("nostack");
    cMCPosACS->BuildLegend();
    cMCPosACS->Print("mcPosACS.png");
    TCanvas* cMCNegACS = new TCanvas();
    mMCNegACS->Draw("nostack");
    cMCNegACS->BuildLegend();
    cMCNegACS->Print("mcNegACS.png");
    TCanvas* cMCDauDCA = new TCanvas();
    mMCDauDCA->Draw("nostack");
    cMCDauDCA->BuildLegend();
    cMCDauDCA->Print("mcDauDCA.png");
    TCanvas* cMCPosPVDCA = new TCanvas();
    mMCPosPVDCA->Draw("nostack");
    cMCPosPVDCA->BuildLegend();
    cMCPosPVDCA->Print("mcPosPVDCA.png");
    TCanvas* cMCNegPVDCA = new TCanvas();
    mMCNegPVDCA->Draw("nostack");
    cMCNegPVDCA->BuildLegend();
    cMCNegPVDCA->Print("mcNegPVDCA.png");
    TCanvas* cMCV0PVDCA = new TCanvas();
    mMCV0PVDCA->Draw("nostack");
    cMCV0PVDCA->BuildLegend();
    cMCV0PVDCA->Print("mcV0PVDCA.png");
    TCanvas* cMCArmPodolTg = new TCanvas();
    mMCArmPodolisTg->Draw("COLZ");
    cMCArmPodolTg->Print("mcArmPodolTg.png");
    TCanvas* cMCArmPodolBg = new TCanvas();
    mMCArmPodolisBg->Draw("COLZ");
    cMCArmPodolBg->Print("mcArmPodolBg.png");
  }

  TCanvas* c10 = new TCanvas();
  mAvgClusSizeCEta->Draw("P NOSTACK");
  c10->BuildLegend(0.6, 0.6, 0.8, 0.8);
  c10->Print("clusSizeEta.png");
}

void AvgClusSizeStudy::endOfStream(EndOfStreamContext& ec)
{
  auto& params = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
  setStyle();
  saveHistograms();
  if (params.generatePlots) {
    plotHistograms();
  }
}

void AvgClusSizeStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
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

DataProcessorSpec getAvgClusSizeStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestClusters(srcClustersMask, useMC);
  dataRequest->requestSecondaryVertices(useMC);
  dataRequest->requestPrimaryVertertices(useMC); // NOTE: may be necessary to use requestPrimaryVerterticesTMP()...
  // dataRequest->requestPrimaryVerterticesTMP(useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              false,                             // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "its-study-AvgClusSize",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<AvgClusSizeStudy>(dataRequest, ggRequest, useMC, kineReader)},
    Options{}};
}
} // namespace study
} // namespace its
} // namespace o2
