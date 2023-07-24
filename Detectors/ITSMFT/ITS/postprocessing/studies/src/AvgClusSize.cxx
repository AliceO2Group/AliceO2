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

#include "DataFormatsITS/TrackITS.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITSBase/GeometryTGeo.h"
#include <SimulationDataFormat/MCTrack.h>
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/DCA.h"
#include "Steer/MCKinematicsReader.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ITStracking/IOUtils.h"

#include <TH1F.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TFitResultPtr.h>
#include <TFitResult.h>
#include <TLine.h>
#include <numeric>
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
  void getClusterSizes(std::vector<int>&, const gsl::span<const o2::itsmft::CompClusterExt>, gsl::span<const unsigned char>::iterator&, const o2::itsmft::TopologyDictionary*);
  void fitMassSpectrum();
  void saveHistograms();
  void plotHistograms();
  void fillEtaBin(float eta, float clusSize, int i);

  // Running options
  bool mUseMC;

  // Data
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<int> mInputClusterSizes;
  gsl::span<const int> mInputITSidxs;
  std::vector<o2::MCTrack> mMCTracks;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;

  // Output plots
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::unique_ptr<TNtuple> mOutputNtuple;

  std::unique_ptr<THStack> mMassSpectrumFull{};
  std::unique_ptr<TH1F> mMassSpectrumFullNC{};
  std::unique_ptr<TH1F> mMassSpectrumFullC{};
  std::unique_ptr<THStack> mMassSpectrumK0s{};
  std::unique_ptr<TH1F> mMassSpectrumK0sNC{};
  std::unique_ptr<TH1F> mMassSpectrumK0sC{};
  std::unique_ptr<THStack> mAvgClusSize{};
  std::unique_ptr<TH1F> mAvgClusSizeNC{};
  std::unique_ptr<TH1F> mAvgClusSizeC{};
  std::unique_ptr<THStack> mAvgClusSizeCEta{};
  std::vector<std::unique_ptr<TH1F>> mAvgClusSizeCEtaVec{};
  std::unique_ptr<THStack> mMCStackCosPA{};
  std::unique_ptr<THStack> mStackDCA{};
  std::unique_ptr<THStack> mStackR{};
  std::unique_ptr<THStack> mStackPVDCA{};
  std::unique_ptr<TH1F> mCosPA{};
  std::unique_ptr<TH1F> mMCCosPAK0{};
  std::unique_ptr<TH1F> mMCCosPAnotK0{};
  std::unique_ptr<TH1F> mCosPAtrueK0{};
  std::unique_ptr<TH1F> mR{};
  std::unique_ptr<TH1F> mRK0{};
  std::unique_ptr<TH1F> mRnotK0{};
  std::unique_ptr<TH1F> mRtrueK0{};
  std::unique_ptr<TH1F> mDCA{};
  std::unique_ptr<TH1F> mDCAK0{};
  std::unique_ptr<TH1F> mDCAnotK0{};
  std::unique_ptr<TH1F> mDCAtrueK0{};
  std::unique_ptr<TH1F> mEtaNC{};
  std::unique_ptr<TH1F> mEtaC{};
  std::unique_ptr<TH1F> mMCMotherPDG{};
  std::unique_ptr<TH1F> mPVDCAK0{};
  std::unique_ptr<TH1F> mPVDCAnotK0{};

  int globalNClusters = 0;
  int globalNPixels = 0;

  std::vector<float> mEtaBinUL; // upper edges for eta bins

  // Counters for K0s identification
  int nNotValid = 0;
  int nNullptrs = 0;
  int nPiPi = 0;
  int nIsPiPiNotK0s = 0;
  int nIsPiPiIsK0s = 0;
  int nIsNotPiPiIsK0s = 0;
  int nMotherIDMismatch = 0;
  int nEvIDMismatch = 0;
  int nK0s = 0;
  int nNotK0s = 0;
  int nPionsInEtaRange = 0;
  int nInvalidK0sMother = 0;

  const std::string mOutName{"o2standalone_cluster_size_study.root"};
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

  LOGP(important, "Cluster size study initialized.");
}

void AvgClusSizeStudy::prepareOutput()
{
  auto& params = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
  mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mOutName.c_str(), "recreate");
  mOutputNtuple = std::make_unique<TNtuple>("v0_data", "v0 data", "d0ACS:d1ACS:mass:d01dca:cosPA:V0R:eta:d0pvDCA:d1pvDCA:isMCK0s");

  mMassSpectrumFull = std::make_unique<THStack>("V0", "V0 mass spectrum; MeV");                                        // auto-set axis ranges
  mMassSpectrumFullNC = std::make_unique<TH1F>("V0nc", "no cuts; MeV", 100, 250, 1000);                                // auto-set axis ranges
  mMassSpectrumFullC = std::make_unique<TH1F>("V0c", "cut; MeV", 100, 250, 1000);                                      // auto-set axis ranges
  mMassSpectrumK0s = std::make_unique<THStack>("K0s", "'K0' mass spectrum; MeV");                                      // set axis ranges near K0short mass
  mMassSpectrumK0sNC = std::make_unique<TH1F>("K0snc", "no cuts; MeV", 15, 475, 525);                                  // set axis ranges near K0short mass
  mMassSpectrumK0sC = std::make_unique<TH1F>("K0sc", "cut; MeV", 15, 475, 525);                                        // set axis ranges near K0short mass
  mAvgClusSize = std::make_unique<THStack>("avgclussize", "Average cluster size per track; pixels / cluster / track"); // auto-set axis ranges
  mAvgClusSizeNC = std::make_unique<TH1F>("avgclussizeNC", "no cuts", 40, 0, 15);                                      // auto-set axis ranges
  mAvgClusSizeC = std::make_unique<TH1F>("avgclussizeC", "cut", 40, 0, 15);                                            // auto-set axis ranges
  mMCStackCosPA = std::make_unique<THStack>("CosPAstack", "CosPA");                                                    // auto-set axis ranges
  mStackDCA = std::make_unique<THStack>("DCAstack", "DCA");                                                            // auto-set axis ranges
  mStackR = std::make_unique<THStack>("Rstack", "R");                                                                  // auto-set axis ranges
  mStackPVDCA = std::make_unique<THStack>("PVDCAstack", "PV-DCA");                                                     // auto-set axis ranges
  mCosPA = std::make_unique<TH1F>("CosPA", "cos(PA)", 100, -1, 1);                                                     // auto-set axis ranges
  mMCCosPAK0 = std::make_unique<TH1F>("CosPAK0", "cos(PA)", 100, -1, 1);                                               // auto-set axis ranges
  mMCCosPAnotK0 = std::make_unique<TH1F>("CosPAnotK0", "cos(PA)", 100, -1, 1);                                         // auto-set axis ranges
  mR = std::make_unique<TH1F>("R", "R", 40, 1, -1);                                                                    // auto-set axis ranges
  mRK0 = std::make_unique<TH1F>("RK0", "R", 40, 0, 20);                                                                // auto-set axis ranges
  mRnotK0 = std::make_unique<TH1F>("RnotK0", "R", 40, 0, 20);                                                          // auto-set axis ranges
  mDCA = std::make_unique<TH1F>("DCA", "DCA", 40, 1, -1);                                                              // auto-set axis ranges
  mDCAK0 = std::make_unique<TH1F>("DCAK0", "DCA", 40, 0, 0.25);                                                        // auto-set axis ranges
  mDCAnotK0 = std::make_unique<TH1F>("DCAnotK0", "DCA", 40, 0, 0.25);                                                  // auto-set axis ranges
  mEtaNC = std::make_unique<TH1F>("etaNC", "no cuts", 30, 1, -1);                                                      // auto-set axis ranges
  mEtaC = std::make_unique<TH1F>("etaC", "cut", 30, 1, -1);                                                            // auto-set axis ranges
  mMCMotherPDG = std::make_unique<TH1F>("PID", "MC K0s mother PDG codes", 100, 1, -1);
  mPVDCAK0 = std::make_unique<TH1F>("PVDCAK0", "Prong DCA to pVertex", 80, 0, 2);
  mPVDCAnotK0 = std::make_unique<TH1F>("PVDCAnotK0", "Prong DCA to pVertex", 80, 0, 2);

  mAvgClusSizeCEta = std::make_unique<THStack>("avgclussizeeta", "Average cluster size per track; pixels / cluster / track"); // auto-set axis ranges
  float binWidth = (params.etaMax - params.etaMin) / (float)params.etaNBins;
  mEtaBinUL.reserve(params.etaNBins);
  for (int i = 0; i < params.etaNBins; i++) {
    mEtaBinUL.emplace_back(params.etaMin + (binWidth * (i + 1)));
    mAvgClusSizeCEtaVec.push_back(std::make_unique<TH1F>(Form("avgclussize%i", i), Form("%.2f < #eta < %.2f", mEtaBinUL[i] - binWidth, mEtaBinUL[i]), params.sizeNBins, 0, params.sizeMax));
    mAvgClusSizeCEtaVec[i]->SetDirectory(nullptr);
    mAvgClusSizeCEta->Add(mAvgClusSizeCEtaVec[i].get());
  }

  mMassSpectrumFullNC->SetDirectory(nullptr);
  mMassSpectrumFullC->SetDirectory(nullptr);
  mMassSpectrumK0sNC->SetDirectory(nullptr);
  mMassSpectrumK0sC->SetDirectory(nullptr);
  mAvgClusSizeNC->SetDirectory(nullptr);
  mAvgClusSizeC->SetDirectory(nullptr);
  mCosPA->SetDirectory(nullptr);
  mMCCosPAK0->SetDirectory(nullptr);
  mMCCosPAnotK0->SetDirectory(nullptr);
  mR->SetDirectory(nullptr);
  mRK0->SetDirectory(nullptr);
  mRnotK0->SetDirectory(nullptr);
  mDCA->SetDirectory(nullptr);
  mDCAK0->SetDirectory(nullptr);
  mDCAnotK0->SetDirectory(nullptr);
  mEtaNC->SetDirectory(nullptr);
  mEtaC->SetDirectory(nullptr);
  mMCMotherPDG->SetDirectory(nullptr);
  mPVDCAK0->SetDirectory(nullptr);
  mPVDCAnotK0->SetDirectory(nullptr);

  mOutputNtuple->SetDirectory(nullptr);

  mMassSpectrumFull->Add(mMassSpectrumFullC.get());
  mMassSpectrumFull->Add(mMassSpectrumFullNC.get());
  mMassSpectrumK0s->Add(mMassSpectrumK0sC.get());
  mMassSpectrumK0s->Add(mMassSpectrumK0sNC.get());
  mAvgClusSize->Add(mAvgClusSizeC.get());
  mAvgClusSize->Add(mAvgClusSizeNC.get());
  mMCStackCosPA->Add(mMCCosPAK0.get());
  mMCStackCosPA->Add(mMCCosPAnotK0.get());
  mStackDCA->Add(mDCAK0.get());
  mStackDCA->Add(mDCAnotK0.get());
  mStackR->Add(mRK0.get());
  mStackR->Add(mRnotK0.get());
  mStackPVDCA->Add(mPVDCAK0.get());
  mStackPVDCA->Add(mPVDCAnotK0.get());
}

void AvgClusSizeStudy::setStyle()
{
  gStyle->SetPalette(kRainbow);

  for (int i = 0; i < mAvgClusSizeCEtaVec.size(); i++) {
    mAvgClusSizeCEtaVec[i]->SetMarkerStyle(20 + i);
    mAvgClusSizeCEtaVec[i]->SetMarkerColor(i + 1);
    mAvgClusSizeCEtaVec[i]->SetLineColor(i + 1);
  }

  mAvgClusSizeC->SetLineColor(kRed);
  mMassSpectrumFullC->SetLineColor(kRed);
  mMassSpectrumK0sC->SetLineColor(kRed);
  mMCCosPAK0->SetLineColor(kRed);
  mDCAK0->SetLineColor(kRed);
  mRK0->SetLineColor(kRed);
  mPVDCAK0->SetLineColor(kRed);
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
  // auto pattIt = clusPatt.begin(); // NOTE: possibly these are not needed; looks like it just finds the 3D spacepoint of the cluster?
  // std::vector<ITSCluster> inputITSclusters;
  // inputITSclusters.reserve(compClus.size());
  // o2::its::ioutils::convertCompactClusters(compClus, pattIt, inputITSclusters, mDict);
  mInputClusterSizes.resize(compClus.size());
  auto pattIt2 = clusPatt.begin();
  getClusterSizes(mInputClusterSizes, compClus, pattIt2, mDict);
}

void AvgClusSizeStudy::process(o2::globaltracking::RecoContainer& recoData)
{
  auto& params = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
  float d0ACS, d1ACS, v0InvMass, d01DCA, cosPA, v0R, eta, d0pvDCA, d1pvDCA;
  bool isMCK0s = false;
  TrackITS d0recoTrk, d1recoTrk; // daughter ITS tracks
  DCA d0DCA, d1DCA;              // DCA object for prong to primary vertex (DCA = o2::dataformats::DCA)
  float b = 5.;                  // Magnetic field in kG (?)
  PVertex pv;

  // Variables for MC analysis
  int totalK0sInDataset = 0;
  gsl::span<const o2::MCCompLabel> mcLabels;
  o2::MCCompLabel d0lab, d1lab;
  const o2::MCTrack *V0mcTrk, *d0mcTrk, *d1mcTrk;
  int V0PdgCode, d0PdgCode, d1PdgCode, m0TrkId, m1TrkId;

  loadData(recoData);
  auto V0s = recoData.getV0s();
  auto trks = recoData.getITSTracks();

  LOGP(info, "Found {} reconstructed V0s.", V0s.size());
  LOGP(info, "Found {} ITS tracks.", trks.size());
  LOGP(info, "Found {} ROFs.", recoData.getITSTracksROFRecords().size());
  if (mUseMC) {
    mcLabels = recoData.getITSTracksMCLabels();
    LOGP(info, "Found {} labels.", mcLabels.size());
  }

  for (auto& v0 : V0s) {
    d0recoTrk = recoData.getITSTrack(v0.getProngID(0));
    d1recoTrk = recoData.getITSTrack(v0.getProngID(1));

    pv = recoData.getPrimaryVertex(v0.getVertexID()); // extract primary vertex
    d0recoTrk.propagateToDCA(pv, b, &d0DCA);          // calculate and store DCA objects for both prongs
    d1recoTrk.propagateToDCA(pv, b, &d1DCA);
    d0pvDCA = std::sqrt(d0DCA.getR2()); // calculate DCA distance
    d1pvDCA = std::sqrt(d1DCA.getR2());

    eta = v0.getEta();
    d01DCA = v0.getDCA();
    cosPA = v0.getCosPA();
    v0InvMass = std::sqrt(v0.calcMass2()) * 1000; // convert mass to MeV
    v0R = std::sqrt(v0.calcR2());                 // gives distance from pvertex to origin? in centimeters (?) NOTE: unsure if this is to the primary vertex or to origin
    if (mUseMC) {                                 // check whether V0 is a K0s in MC, and fill the cut validation plots
      isMCK0s = false;
      d0lab = mcLabels[v0.getProngID(0)]; // extract MC label for the prongs
      d1lab = mcLabels[v0.getProngID(1)];
      // Now we check validity, etc. for the labels (essentially strength of reco) to determine which reconstructed V0 are real K0s
      if (d0lab.isValid() && d1lab.isValid()) {
        d0mcTrk = mKineReader->getTrack(d0lab);
        LOGP(debug, "Got d0 track");
        d1mcTrk = mKineReader->getTrack(d1lab);
        LOGP(debug, "Got d1 track");
        if (d0mcTrk == nullptr || d1mcTrk == nullptr) {
          LOGP(debug, "Nullptr found, skipping this V0");
          nNullptrs++;
        } else {
          LOGP(debug, "About to query Pdg codes");
          d0PdgCode = d0mcTrk->GetPdgCode();
          d1PdgCode = d1mcTrk->GetPdgCode();
          m0TrkId = d0mcTrk->getMotherTrackId();
          m1TrkId = d1mcTrk->getMotherTrackId();
          LOGP(debug, "pdgcodes are {} and {}", d0PdgCode, d1PdgCode);

          if (m0TrkId == m1TrkId && m0TrkId != -1 && m1TrkId != -1) {
            if (d1lab.getEventID() == d0lab.getEventID()) {
              V0mcTrk = mKineReader->getTrack(d1lab.getEventID(), m0TrkId); // assume daughter MCTracks are in same event as V0 MCTrack
              V0PdgCode = V0mcTrk->GetPdgCode();
              if (V0PdgCode == 310) {
                isMCK0s = true;
                nK0s++;
                if (abs(d0PdgCode) == 211 && d0PdgCode / d1PdgCode == -1) {
                  nIsPiPiIsK0s++;
                } else {
                  nIsNotPiPiIsK0s++;
                }
              } else {
                if (abs(d0PdgCode) == 211 && d0PdgCode / d1PdgCode == -1) {
                  nIsPiPiNotK0s++;
                }
                nNotK0s++;
              }
              if (abs(d0PdgCode) == 211 && d0PdgCode / d1PdgCode == -1) {
                nPiPi++;
              }
            } else {
              nEvIDMismatch++;
            }
          } else {
            nMotherIDMismatch++;
          }
        }
      } else {
        nNotValid++;
      }
      if (isMCK0s) {
        mMCCosPAK0->Fill(cosPA);
        mDCAK0->Fill(d01DCA);
        mRK0->Fill(v0R);
        mPVDCAK0->Fill(d0pvDCA);
        mPVDCAK0->Fill(d1pvDCA);
      } else {
        mMCCosPAnotK0->Fill(cosPA);
        mDCAnotK0->Fill(d01DCA);
        mRnotK0->Fill(v0R);
        mPVDCAnotK0->Fill(d0pvDCA);
        mPVDCAnotK0->Fill(d1pvDCA);
      }
    }

    d0ACS = getAverageClusterSize(&d0recoTrk);
    d1ACS = getAverageClusterSize(&d1recoTrk);
    mAvgClusSizeNC->Fill(d0ACS);
    mAvgClusSizeNC->Fill(d1ACS);

    mOutputNtuple->Fill(d0ACS, d1ACS, v0InvMass, d01DCA, cosPA, v0R, eta, d0pvDCA, d1pvDCA, (float)isMCK0s);

    mCosPA->Fill(cosPA);
    mMassSpectrumFullNC->Fill(v0InvMass);
    mMassSpectrumK0sNC->Fill(v0InvMass);
    mEtaNC->Fill(eta);
    mR->Fill(v0R);
    mDCA->Fill(d01DCA);
    // innermost layer of ITS lies at 2.3cm
    if (cosPA > params.cosPAmin && v0R < params.Rmax && v0R > params.Rmin && d01DCA < params.prongDCAmax && d0pvDCA > params.dauPVDCAmin && d1pvDCA > params.dauPVDCAmin) {
      mMassSpectrumK0sC->Fill(v0InvMass);
      mMassSpectrumFullC->Fill(v0InvMass);
      mAvgClusSizeC->Fill(d0ACS);
      mAvgClusSizeC->Fill(d1ACS);
      if (eta > params.etaMin && eta < params.etaMax) {
        nPionsInEtaRange++;
        fillEtaBin(eta, d0ACS, 0);
        fillEtaBin(eta, d1ACS, 0);
      }
      mEtaC->Fill(eta);
    }
  }

  if (mUseMC) {
    o2::MCTrack V0MotherMCTrk;
    LOGP(info, "OVERALL STATISTICS: {} nonvalid daughter pairs, {} nullptrs, {} motherID mismatches, {} evID mismatches, {} K0-shorts, {} not-K0s, {} pion pairs, out of {} V0s", nNotValid, nNullptrs, nMotherIDMismatch, nEvIDMismatch, nK0s, nNotK0s, nPiPi, V0s.size());
    LOGP(info, "OVERALL STATISTICS: {} Pi Y K0s N, {} Pi Y K0s Y, {} Pi N K0s Y", nIsPiPiNotK0s, nIsPiPiIsK0s, nIsNotPiPiIsK0s);
    LOGP(info, "OVERALL STATISTICS: {} Pions in eta range", nPionsInEtaRange);
    int nK0sisPrimary = 0;
    int totalK0sMotherMinus1 = 0;
    for (auto mcTrk : mMCTracks) { // search through all MC tracks to find K0s, whether reconstructed or not
      if (mcTrk.GetPdgCode() == 310 && mcTrk.getMotherTrackId() == -1) {
        totalK0sMotherMinus1++;
      }
      if (mcTrk.GetPdgCode() == 310 && mcTrk.isPrimary()) {
        nK0sisPrimary++;
        // V0MotherMCTrk = mMCTracks[mcTrk.getMotherTrackId()];
        // mMCMotherPDG->Fill(V0MotherMCTrk.GetPdgCode());
        // LOGP(info, "K0s is primary, motherID is {} and motherPDG is {}", mcTrk.getMotherTrackId(), V0MotherMCTrk.GetPdgCode());
      }
    }
    LOGP(info, "OVERALL STATISTICS: {} K0s (mother==-1) found in MC tracks out of {} total", totalK0sMotherMinus1, mMCTracks.size());
    LOGP(info, "OVERALL STATISTICS: {} K0s (isPrimary) found in MC tracks out of {} total", nK0sisPrimary, mMCTracks.size());
    // LOGP(info, "OVERALL STATISTICS: {} primary K0s found in MC tracks out of {} total", totalK0sInDataset, mMCTracks.size());
  }

  // TODO: implement 7 cluster track cut for daughters; if we don't have enough statistics, we can cut even harsher on cosPA and inject more statistics
  // TODO: print the cut on the graph
}

float AvgClusSizeStudy::getAverageClusterSize(TrackITS* daughter)
{
  int totalSize{0};
  auto firstClus = daughter->getFirstClusterEntry();
  auto ncl = daughter->getNumberOfClusters();
  for (int icl = 0; icl < ncl; icl++) {
    totalSize += mInputClusterSizes[mInputITSidxs[firstClus + icl]];
    globalNPixels += mInputClusterSizes[mInputITSidxs[firstClus + icl]];
  }
  globalNClusters += ncl;
  return (float)totalSize / (float)ncl;
}

void AvgClusSizeStudy::fillEtaBin(float eta, float clusSize, int i)
{
  if (eta < mEtaBinUL[i]) { // FIXME: there is a problem if eta is outside the full range (< etaMin or > etaMax)...
    mAvgClusSizeCEtaVec[i]->Fill(clusSize);
  } else {
    fillEtaBin(eta, clusSize, i + 1);
  }
}

void AvgClusSizeStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void AvgClusSizeStudy::saveHistograms()
{
  mDBGOut.reset();
  TFile fout(mOutName.c_str(), "RECREATE");

  fout.WriteTObject(mOutputNtuple.get());

  fout.WriteTObject(mMassSpectrumFullNC.get());
  fout.WriteTObject(mMassSpectrumFullC.get());
  fout.WriteTObject(mMassSpectrumK0sNC.get());
  fout.WriteTObject(mMassSpectrumK0sC.get());
  fout.WriteTObject(mAvgClusSize.get());
  fout.WriteTObject(mAvgClusSizeCEta.get()); // NOTE: storing the THStack does work, but it's a little more complicated to extract the individual histograms
  fout.WriteTObject(mCosPA.get());
  fout.WriteTObject(mR.get());
  fout.WriteTObject(mDCA.get());
  fout.WriteTObject(mEtaNC.get());
  fout.WriteTObject(mEtaC.get());
  fout.WriteTObject(mMCMotherPDG.get());
  fout.WriteTObject(mMCCosPAK0.get());
  fout.WriteTObject(mMCCosPAnotK0.get());
  fout.WriteTObject(mRK0.get());
  fout.WriteTObject(mRnotK0.get());
  fout.WriteTObject(mDCAK0.get());
  fout.WriteTObject(mDCAnotK0.get());
  fout.WriteTObject(mPVDCAK0.get());
  fout.WriteTObject(mPVDCAnotK0.get());
  fout.Close();

  // mMCStackCosPA->Add(mMCCosPAK0.get());
  // mMCStackCosPA->Add(mMCCosPAnotK0.get());
  // mStackDCA->Add(mDCAK0.get());
  // mStackDCA->Add(mDCAnotK0.get());
  // mStackR->Add(mRK0.get());
  // mStackR->Add(mRnotK0.get());

  LOGP(important, "Stored histograms into {}", mOutName.c_str());
}

void AvgClusSizeStudy::plotHistograms()
{
  auto& params = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
  float globalAvgClusSize = (float)globalNPixels / (float)globalNClusters;

  TCanvas* c1 = new TCanvas();
  mMassSpectrumFull->Draw("nostack");
  c1->Print("massSpectrumFull.png");

  TCanvas* c2 = new TCanvas();
  mMassSpectrumK0s->Draw("nostack E");
  c2->BuildLegend();
  c2->Print("massSpectrumK0s.png");

  TCanvas* c3 = new TCanvas();
  mAvgClusSize->Draw("nostack");
  TLine* globalAvg = new TLine(globalAvgClusSize, 0, globalAvgClusSize, mAvgClusSizeNC->GetMaximum());
  globalAvg->Draw();
  c3->Print("clusSize.png");

  TCanvas* c4 = new TCanvas();
  mCosPA->Draw();
  c4->Print("cosPA.png");

  TCanvas* c6 = new TCanvas();
  mR->Draw();
  c6->Print("mR.png");

  TCanvas* c7 = new TCanvas();
  mDCA->Draw();
  c7->Print("mDCA.png");

  TCanvas* c8 = new TCanvas();
  mEtaNC->Draw();
  c8->Print("mEtaNC.png");

  TCanvas* c9 = new TCanvas();
  mEtaC->Draw();
  c9->Print("mEtaC.png");

  TCanvas* c10 = new TCanvas();
  mAvgClusSizeCEta->Draw("P NOSTACK");
  c10->BuildLegend(0.6, 0.6, 0.8, 0.8);
  c10->Print("clusSizeEta.png");
  for (int i = 0; i < params.etaNBins; i++) {
    mAvgClusSizeCEtaVec[i]->Scale(1. / mAvgClusSizeCEtaVec[i]->Integral("width"));
  }

  TCanvas* c11 = new TCanvas();
  mAvgClusSizeCEta->Draw("P L NOSTACK HIST");
  mAvgClusSizeCEta->SetTitle("Average cluster size per track (normed)");
  c11->BuildLegend(0.6, 0.6, 0.8, 0.8);
  c11->Print("clusSizeEtaNormed.png");

  if (mUseMC) {
    TCanvas* c12 = new TCanvas();
    mMCMotherPDG->Draw();
    c12->Print("MCMotherPDG.png");

    TCanvas* c13 = new TCanvas();
    mMCStackCosPA->Draw("nostack");
    c13->Print("MCCosPA.png");

    TCanvas* c14 = new TCanvas();
    mStackDCA->Draw("nostack");
    c14->Print("MCDCA.png");

    TCanvas* c15 = new TCanvas();
    mStackR->Draw("nostack");
    c15->Print("MCR.png");

    TCanvas* c16 = new TCanvas();
    mStackPVDCA->Draw("nostack");
    c16->Print("MCPVDCA.png");
  }
}

void AvgClusSizeStudy::fitMassSpectrum()
{
  TF1* gaus = new TF1("gaus", "gaus", 485, 505);
  TFitResultPtr fit = mMassSpectrumK0sC->Fit("gaus", "S", "", 480, 510);
  fit->Print();
}

void AvgClusSizeStudy::endOfStream(EndOfStreamContext& ec)
{
  auto& params = o2::its::study::ITSAvgClusSizeParamConfig::Instance();
  if (params.performFit) {
    fitMassSpectrum();
  }
  if (params.generatePlots) {
    saveHistograms();
    setStyle();
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