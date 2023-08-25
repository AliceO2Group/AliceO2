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

#include "ITSStudies/TrackCheck.h"
#include "ITSStudies/ITSStudiesConfigParam.h"

#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCTrack.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "CommonUtils/TreeStreamRedirector.h"

#include "Framework/Task.h"
#include "Steer/MCKinematicsReader.h"
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TEfficiency.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TObjArray.h>
#include <THStack.h>

namespace o2
{
namespace its
{
namespace study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;
using o2::steer::MCKinematicsReader;
class TrackCheckStudy : public Task
{
  struct ParticleInfo {
    int event;
    int pdg;
    float pt;
    float eta;
    float phi;
    int mother;
    int first;
    float vx;
    float vy;
    float vz;
    int unsigned short clusters = 0u;
    unsigned char isReco = 0u;
    unsigned char isFake = 0u;
    bool isPrimary = 0u;
    unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
    o2::its::TrackITS track;
  };

 public:
  TrackCheckStudy(std::shared_ptr<DataRequest> dr,
                  mask_t src,
                  bool useMC,
                  std::shared_ptr<o2::steer::MCKinematicsReader> kineReader,
                  std::shared_ptr<o2::base::GRPGeomRequest> gr) : mDataRequest(dr), mTracksSrc(src), mKineReader(kineReader), mGGCCDBRequest(gr)
  {
    if (useMC) {
      LOGP(info, "Read MCKine reader with {} sources", mKineReader->getNSources());
    }
  }

  ~TrackCheckStudy() final = default;
  void init(InitContext&) final;
  void run(ProcessingContext&) final;
  void endOfStream(EndOfStreamContext&) final;
  void finaliseCCDB(ConcreteDataMatcher&, void*) final;
  void initialiseRun(o2::globaltracking::RecoContainer&);
  void process();
  void setEfficiencyGraph(std::unique_ptr<TEfficiency>&, const char*, const int, const double, const double, const int, const double);
  void setEfficiencyGraph(std::unique_ptr<TEfficiency>&, const char*, const char*, const int, const double, const double, const int, const double);
  void setHistoMCGraph(TH1D&, std::unique_ptr<TH1D>&, const char*, const char*, const int, const double);
  void NormalizeHistos(std::vector<TH1D>&);

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  std::string mOutFileName = "TrackCheckStudy.root";
  std::shared_ptr<MCKinematicsReader> mKineReader;
  GeometryTGeo* mGeometry;

  // Spans
  gsl::span<const o2::itsmft::ROFRecord> mTracksROFRecords;
  gsl::span<const o2::its::TrackITS> mTracks;
  gsl::span<const o2::MCCompLabel> mTracksMCLabels;
  gsl::span<const o2::itsmft::CompClusterExt> mClusters;
  const o2::dataformats::MCLabelContainer* mClustersMCLCont;

  // Data
  GTrackID::mask_t mTracksSrc{};
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<std::vector<std::vector<ParticleInfo>>> mParticleInfo; // src/event/track
  std::vector<ParticleInfo> mParticleInfoPrim;
  std::vector<ParticleInfo> mParticleInfoSec;
  unsigned short mMask = 0x7f;

  // Utils
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;

  // Histos
  std::unique_ptr<TH1D> mGoodPt;
  std::unique_ptr<TH1D> mGoodEta;
  std::unique_ptr<TH1D> mGoodChi2;
  std::unique_ptr<TH1D> mFakePt;
  std::unique_ptr<TH1D> mFakeEta;
  std::unique_ptr<TH1D> mMultiFake;
  std::unique_ptr<TH1D> mFakeChi2;
  std::unique_ptr<TH1D> mClonePt;
  std::unique_ptr<TH1D> mCloneEta;

  std::unique_ptr<TH1D> mDenominatorPt;
  std::unique_ptr<TH1D> mDenominatorEta;
  std::unique_ptr<TH1D> mDenominatorSecRad;
  std::unique_ptr<TH1D> mDenominatorSecZ;

  std::unique_ptr<TH1D> mGoodRad; // decay radius and z of sv for secondary particle
  std::unique_ptr<TH1D> mFakeRad;
  std::unique_ptr<TH1D> mGoodZ;
  std::unique_ptr<TH1D> mFakeZ;
  std::unique_ptr<TH1D> mRadk; // decay radius and z of sv for particle with mother k e lambda
  std::unique_ptr<TH1D> mZk;
  std::unique_ptr<TH1D> mRadLam;
  std::unique_ptr<TH1D> mZLam;
  std::unique_ptr<TH1D> mGoodRadk;
  std::unique_ptr<TH1D> mFakeRadk;
  std::unique_ptr<TH1D> mGoodZk;
  std::unique_ptr<TH1D> mFakeZk;
  std::unique_ptr<TH1D> mGoodRadLam;
  std::unique_ptr<TH1D> mFakeRadLam;
  std::unique_ptr<TH1D> mGoodZLam;
  std::unique_ptr<TH1D> mFakeZLam;

  std::unique_ptr<TEfficiency> mEffPt; // Eff vs Pt primary
  std::unique_ptr<TEfficiency> mEffFakePt;
  std::unique_ptr<TEfficiency> mEffClonesPt;
  std::unique_ptr<TEfficiency> mEffEta; // Eff vs Eta primary
  std::unique_ptr<TEfficiency> mEffFakeEta;
  std::unique_ptr<TEfficiency> mEffClonesEta;

  std::unique_ptr<TEfficiency> mEffRad; // Eff vs Radius secondary
  std::unique_ptr<TEfficiency> mEffFakeRad;
  std::unique_ptr<TEfficiency> mEffZ; // Eff vs Z of sv secondary
  std::unique_ptr<TEfficiency> mEffFakeZ;
  std::unique_ptr<TEfficiency> mEffRadk; // Eff vs Z of sv and decay radius secondary for particle with mother k e lambda
  std::unique_ptr<TEfficiency> mEffFakeRadk;
  std::unique_ptr<TEfficiency> mEffZk;
  std::unique_ptr<TEfficiency> mEffFakeZk;
  std::unique_ptr<TEfficiency> mEffRadLam;
  std::unique_ptr<TEfficiency> mEffFakeRadLam;
  std::unique_ptr<TEfficiency> mEffZLam;
  std::unique_ptr<TEfficiency> mEffFakeZLam;

  std::unique_ptr<TH1D> mPtResolution; // Pt resolution for both primary and secondary
  std::unique_ptr<TH2D> mPtResolution2D;
  std::unique_ptr<TH1D> mPtResolutionSec;
  std::unique_ptr<TH1D> mPtResolutionPrim;
  std::unique_ptr<TGraphErrors> g1;

  std::vector<TH1I*> histLength, histLength1Fake, histLength2Fake, histLength3Fake, histLengthNoCl, histLength1FakeNoCl, histLength2FakeNoCl, histLength3FakeNoCl; // FakeCluster Study
  std::vector<THStack*> stackLength, stackLength1Fake, stackLength2Fake, stackLength3Fake;
  std::vector<TLegend*> legends, legends1Fake, legends2Fake, legends3Fake;
  std::vector<std::unique_ptr<TH1D>> mGoodPts, mFakePts, mTotPts, mGoodEtas, mFakeEtas, mTotEtas, mGoodPtsK, mFakePtsK, mTotPtsK, mGoodPtsLam, mFakePtsLam, mTotPtsLam, mGoodPtsIperT, mFakePtsIperT, mTotPtsIperT;
  std::vector<std::unique_ptr<TEfficiency>> mEffGoodPts, mEffFakePts, mEffGoodEtas, mEffFakeEtas, mEffGoodPtsK, mEffFakePtsK, mEffGoodPtsLam, mEffFakePtsLam, mEffGoodPtsIperT, mEffFakePtsIperT;
  // Canvas & decorations
  std::unique_ptr<TCanvas> mCanvasPt;
  std::unique_ptr<TCanvas> mCanvasPt2;
  std::unique_ptr<TCanvas> mCanvasPt2fake;
  std::unique_ptr<TCanvas> mCanvasEta;
  std::unique_ptr<TCanvas> mCanvasRad;
  std::unique_ptr<TCanvas> mCanvasZ;
  std::unique_ptr<TCanvas> mCanvasRadD;
  std::unique_ptr<TCanvas> mCanvasZD;
  std::unique_ptr<TCanvas> mCanvasEta2;
  std::unique_ptr<TCanvas> mCanvasEta2fake;
  std::unique_ptr<TCanvas> mCanvasPtRes;
  std::unique_ptr<TCanvas> mCanvasPtRes2;
  std::unique_ptr<TCanvas> mCanvasPtRes3;
  std::unique_ptr<TCanvas> mCanvasPtRes4;
  std::unique_ptr<TLegend> mLegendPt;
  std::unique_ptr<TLegend> mLegendPt2;
  std::unique_ptr<TLegend> mLegendPt2Fake;
  std::unique_ptr<TLegend> mLegendEta;
  std::unique_ptr<TLegend> mLegendEta2;
  std::unique_ptr<TLegend> mLegendEta2Fake;
  std::unique_ptr<TLegend> mLegendPtRes;
  std::unique_ptr<TLegend> mLegendPtRes2;
  std::unique_ptr<TLegend> mLegendZ;
  std::unique_ptr<TLegend> mLegendRad;
  std::unique_ptr<TLegend> mLegendZD;
  std::unique_ptr<TLegend> mLegendRadD;
  std::vector<TH1D> HistoMC;

  float rLayer0 = 2.34;
  float rLayer1 = 3.15;
  float rLayer2 = 3.93;
  float rLayer3 = 19.605;

  double sigma[100];
  double sigmaerr[100];
  double meanPt[100];
  double aa[100];

  // Debug output tree
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
};

void TrackCheckStudy::NormalizeHistos(std::vector<TH1D>& HistoMC)
{
  int nHist = HistoMC.size();
  for (int jh = 0; jh < nHist; jh++) {
    double tot = HistoMC[jh].Integral();
    if (tot > 0)
      HistoMC[jh].Scale(1. / tot);
  }
}

void TrackCheckStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  auto& pars = o2::its::study::ITSCheckTracksParamConfig::Instance();
  mOutFileName = pars.outFileName;
  mMask = pars.trackLengthMask;

  std::vector<double> xbins;
  xbins.resize(pars.effHistBins + 1);
  double a = std::log(pars.effPtCutHigh / pars.effPtCutLow) / pars.effHistBins;
  for (int i{0}; i <= pars.effHistBins; i++) {
    xbins[i] = pars.effPtCutLow * std::exp(i * a);
  }
  mGoodPt = std::make_unique<TH1D>("goodPt", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
  mGoodEta = std::make_unique<TH1D>("goodEta", ";#eta;Number of tracks", 60, -3, 3);
  mGoodChi2 = std::make_unique<TH1D>("goodChi2", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", 200, 0, 100);

  mGoodRad = std::make_unique<TH1D>("goodRad", ";#Radius [cm];Number of tracks", 100, 0, 25);
  mGoodZ = std::make_unique<TH1D>("goodZ", ";#z of secondary vertex [cm];Number of tracks", 100, -50, 50);
  mGoodRadk = std::make_unique<TH1D>("goodRadk", ";#Radius [cm];Number of tracks", 100, 0, 25);
  mGoodZk = std::make_unique<TH1D>("goodZk", ";#z of secondary vertex [cm];Number of tracks", 100, -50, 50);
  mGoodRadLam = std::make_unique<TH1D>("goodRadLam", ";#Radius [cm];Number of tracks", 100, 0, 25);
  mGoodZLam = std::make_unique<TH1D>("goodZLam", ";#z of secondary vertex [cm];Number of tracks", 100, -50, 50);

  mFakePt = std::make_unique<TH1D>("fakePt", ";#it{p}_{T} (GeV/#it{c});Fak", pars.effHistBins, xbins.data());
  mFakeEta = std::make_unique<TH1D>("fakeEta", ";#eta;Number of tracks", 60, -3, 3);
  mFakeChi2 = std::make_unique<TH1D>("fakeChi2", ";#it{p}_{T} (GeV/#it{c});Fak", 200, 0, 100);

  mFakeRad = std::make_unique<TH1D>("fakeRad", ";#Radius [cm];Number of tracks", 100, 0, 25);
  mFakeZ = std::make_unique<TH1D>("fakeZ", ";#z of secondary vertex [cm];Number of tracks", 100, -50, 50);
  mFakeRadk = std::make_unique<TH1D>("fakeRadK", ";#Radius [cm];Number of tracks", 100, 0, 25);
  mFakeZk = std::make_unique<TH1D>("fakeZk", ";#z of secondary vertex [cm];Number of tracks", 100, -50, 50);
  mFakeRadLam = std::make_unique<TH1D>("fakeRadLam", ";#Radius [cm];Number of tracks", 100, 0, 25);
  mFakeZLam = std::make_unique<TH1D>("fakeZLam", ";#z of secondary vertex [cm];Number of tracks", 100, -50, 50);

  mMultiFake = std::make_unique<TH1D>("multiFake", ";#it{p}_{T} (GeV/#it{c});Fak", pars.effHistBins, xbins.data());

  mClonePt = std::make_unique<TH1D>("clonePt", ";#it{p}_{T} (GeV/#it{c});Clone", pars.effHistBins, xbins.data());
  mCloneEta = std::make_unique<TH1D>("cloneEta", ";#eta;Number of tracks", 60, -3, 3);

  mDenominatorPt = std::make_unique<TH1D>("denominatorPt", ";#it{p}_{T} (GeV/#it{c});Den", pars.effHistBins, xbins.data());
  mDenominatorEta = std::make_unique<TH1D>("denominatorEta", ";#eta;Number of tracks", 60, -3, 3);
  mDenominatorSecRad = std::make_unique<TH1D>("denominatorSecRad", ";Radius [cm];Number of tracks", 100, 0, 25);
  mDenominatorSecZ = std::make_unique<TH1D>("denominatorSecZ", ";z of secondary vertex [cm];Number of tracks", 100, -50, 50);

  mRadk = std::make_unique<TH1D>("mRadk", ";Radius [cm];Number of tracks", 100, 0, 25);
  mRadLam = std::make_unique<TH1D>("mRadLam", ";Radius [cm];Number of tracks", 100, 0, 25);
  mZk = std::make_unique<TH1D>("mZk", ";z of secondary vertex [cm]", 100, -50, 50);
  mZLam = std::make_unique<TH1D>("mZLam", ";z of secondary vertex [cm]", 100, -50, 50);
  mGoodPts.resize(4);
  mFakePts.resize(4);
  mTotPts.resize(4);
  mGoodEtas.resize(4);
  mFakeEtas.resize(4);
  mTotEtas.resize(4);
  mGoodPtsK.resize(4);
  mFakePtsK.resize(4);
  mTotPtsK.resize(4);
  mGoodPtsLam.resize(4);
  mFakePtsLam.resize(4);
  mTotPtsLam.resize(4);
  mGoodPtsIperT.resize(4);
  mFakePtsIperT.resize(4);
  mTotPtsIperT.resize(4);
  mEffGoodPts.resize(4);
  mEffFakePts.resize(4);
  mEffGoodEtas.resize(4);
  mEffFakeEtas.resize(4);
  mEffGoodPtsK.resize(4);
  mEffFakePtsK.resize(4);
  mEffGoodPtsLam.resize(4);
  mEffFakePtsLam.resize(4);
  mEffGoodPtsIperT.resize(4);
  mEffFakePtsIperT.resize(4);

  for (int yy = 0; yy < 4; yy++) { // divided by layer
    mGoodPts[yy] = std::make_unique<TH1D>(Form("goodPts%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mFakePts[yy] = std::make_unique<TH1D>(Form("FakePs%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mTotPts[yy] = std::make_unique<TH1D>(Form("TotPts%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());

    mGoodEtas[yy] = std::make_unique<TH1D>(Form("goodEtas%d", yy), ";#eta;Number of tracks", 60, -3, 3);
    mFakeEtas[yy] = std::make_unique<TH1D>(Form("FakeEtas%d", yy), ";#eta;Number of tracks", 60, -3, 3);
    mTotEtas[yy] = std::make_unique<TH1D>(Form("TotEtas%d", yy), ";#eta;Number of tracks", 60, -3, 3);

    mGoodPtsK[yy] = std::make_unique<TH1D>(Form("goodPtK%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mFakePtsK[yy] = std::make_unique<TH1D>(Form("FakePtK%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mTotPtsK[yy] = std::make_unique<TH1D>(Form("TotPtK%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mGoodPtsLam[yy] = std::make_unique<TH1D>(Form("goodPtLam%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mFakePtsLam[yy] = std::make_unique<TH1D>(Form("FakePtLam%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mTotPtsLam[yy] = std::make_unique<TH1D>(Form("TotPtLam%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mGoodPtsIperT[yy] = std::make_unique<TH1D>(Form("goodPtIperT%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mFakePtsIperT[yy] = std::make_unique<TH1D>(Form("FakePtIperT%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
    mTotPtsIperT[yy] = std::make_unique<TH1D>(Form("TotPtIperT%d", yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());

    mGoodPts[yy]->Sumw2();
    mFakePts[yy]->Sumw2();
    mTotPts[yy]->Sumw2();
    mGoodEtas[yy]->Sumw2();
    mFakeEtas[yy]->Sumw2();
    mTotEtas[yy]->Sumw2();
    mGoodPtsK[yy]->Sumw2();
    mFakePtsK[yy]->Sumw2();
    mTotPtsK[yy]->Sumw2();
    mGoodPtsLam[yy]->Sumw2();
    mFakePtsLam[yy]->Sumw2();
    mTotPtsLam[yy]->Sumw2();
    mGoodPtsIperT[yy]->Sumw2();
    mFakePtsIperT[yy]->Sumw2();
    mTotPtsIperT[yy]->Sumw2();
  }

  mPtResolution = std::make_unique<TH1D>("PtResolution", ";#it{p}_{T} ;Den", 100, -1, 1);
  mPtResolutionSec = std::make_unique<TH1D>("PtResolutionSec", ";#it{p}_{T} ;Den", 100, -1, 1);
  mPtResolutionPrim = std::make_unique<TH1D>("PtResolutionPrim", ";#it{p}_{T} ;Den", 100, -1, 1);
  mPtResolution2D = std::make_unique<TH2D>("#it{p}_{T} Resolution vs #it{p}_{T}", ";#it{p}_{T} (GeV/#it{c});#Delta p_{T}/p_{T_{MC}", 100, 0, 10, 100, -1, 1);

  mPtResolution->Sumw2();
  mPtResolutionSec->Sumw2();
  mPtResolutionPrim->Sumw2();

  mRadk->Sumw2();
  mRadLam->Sumw2();
  mZk->Sumw2();
  mZLam->Sumw2();

  mGoodPt->Sumw2();
  mGoodEta->Sumw2();
  mGoodRad->Sumw2();
  mGoodZ->Sumw2();
  mGoodRadk->Sumw2();
  mGoodZk->Sumw2();
  mGoodRadLam->Sumw2();
  mGoodZLam->Sumw2();
  mFakeRadk->Sumw2();
  mFakeZk->Sumw2();
  mFakeRadLam->Sumw2();
  mFakeZLam->Sumw2();
  mFakePt->Sumw2();
  mFakeRad->Sumw2();
  mFakeZ->Sumw2();
  mMultiFake->Sumw2();
  mClonePt->Sumw2();
  mDenominatorPt->Sumw2();
  mDenominatorSecRad->Sumw2();
  mDenominatorSecZ->Sumw2();

  histLength.resize(4); // fake clusters study
  histLength1Fake.resize(4);
  histLength2Fake.resize(4);
  histLength3Fake.resize(4);
  histLengthNoCl.resize(4);
  histLength1FakeNoCl.resize(4);
  histLength2FakeNoCl.resize(4);
  histLength3FakeNoCl.resize(4);
  stackLength.resize(4);
  stackLength1Fake.resize(4);
  stackLength2Fake.resize(4);
  stackLength3Fake.resize(4);
  legends.resize(4);
  legends1Fake.resize(4);
  legends2Fake.resize(4);
  legends3Fake.resize(4);

  for (int iH{4}; iH < 8; ++iH) {
    histLength[iH - 4] = new TH1I(Form("trk_len_%d", iH), "#exists cluster", 7, -.5, 6.5);
    histLength[iH - 4]->SetFillColor(kGreen + 3);
    histLength[iH - 4]->SetLineColor(kGreen + 3);
    histLength[iH - 4]->SetFillStyle(3352);
    histLengthNoCl[iH - 4] = new TH1I(Form("trk_len_%d_nocl", iH), "#slash{#exists} cluster", 7, -.5, 6.5);
    histLengthNoCl[iH - 4]->SetFillColor(kOrange + 7);
    histLengthNoCl[iH - 4]->SetLineColor(kOrange + 7);
    histLengthNoCl[iH - 4]->SetFillStyle(3352);
    stackLength[iH - 4] = new THStack(Form("stack_trk_len_%d", iH), Form("trk_len=%d", iH));
    stackLength[iH - 4]->Add(histLength[iH - 4]);
    stackLength[iH - 4]->Add(histLengthNoCl[iH - 4]);

    histLength1Fake[iH - 4] = new TH1I(Form("trk_len_%d_1f", iH), "#exists cluster", 7, -.5, 6.5);
    histLength1Fake[iH - 4]->SetFillColor(kGreen + 3);
    histLength1Fake[iH - 4]->SetLineColor(kGreen + 3);
    histLength1Fake[iH - 4]->SetFillStyle(3352);
    histLength1FakeNoCl[iH - 4] = new TH1I(Form("trk_len_%d_1f_nocl", iH), "#slash{#exists} cluster", 7, -.5, 6.5);
    histLength1FakeNoCl[iH - 4]->SetFillColor(kOrange + 7);
    histLength1FakeNoCl[iH - 4]->SetLineColor(kOrange + 7);
    histLength1FakeNoCl[iH - 4]->SetFillStyle(3352);
    stackLength1Fake[iH - 4] = new THStack(Form("stack_trk_len_%d_1f", iH), Form("trk_len=%d, 1 Fake", iH));
    stackLength1Fake[iH - 4]->Add(histLength1Fake[iH - 4]);
    stackLength1Fake[iH - 4]->Add(histLength1FakeNoCl[iH - 4]);

    histLength2Fake[iH - 4] = new TH1I(Form("trk_len_%d_2f", iH), "#exists cluster", 7, -.5, 6.5);
    histLength2Fake[iH - 4]->SetFillColor(kGreen + 3);
    histLength2Fake[iH - 4]->SetLineColor(kGreen + 3);
    histLength2Fake[iH - 4]->SetFillStyle(3352);
    histLength2FakeNoCl[iH - 4] = new TH1I(Form("trk_len_%d_2f_nocl", iH), "#slash{#exists} cluster", 7, -.5, 6.5);
    histLength2FakeNoCl[iH - 4]->SetFillColor(kOrange + 7);
    histLength2FakeNoCl[iH - 4]->SetLineColor(kOrange + 7);
    histLength2FakeNoCl[iH - 4]->SetFillStyle(3352);
    stackLength2Fake[iH - 4] = new THStack(Form("stack_trk_len_%d_2f", iH), Form("trk_len=%d, 2 Fake", iH));
    stackLength2Fake[iH - 4]->Add(histLength2Fake[iH - 4]);
    stackLength2Fake[iH - 4]->Add(histLength2FakeNoCl[iH - 4]);

    histLength3Fake[iH - 4] = new TH1I(Form("trk_len_%d_3f", iH), "#exists cluster", 7, -.5, 6.5);
    histLength3Fake[iH - 4]->SetFillColor(kGreen + 3);
    histLength3Fake[iH - 4]->SetLineColor(kGreen + 3);
    histLength3Fake[iH - 4]->SetFillStyle(3352);
    histLength3FakeNoCl[iH - 4] = new TH1I(Form("trk_len_%d_3f_nocl", iH), "#slash{#exists} cluster", 7, -.5, 6.5);
    histLength3FakeNoCl[iH - 4]->SetFillColor(kOrange + 7);
    histLength3FakeNoCl[iH - 4]->SetLineColor(kOrange + 7);
    histLength3FakeNoCl[iH - 4]->SetFillStyle(3352);
    stackLength3Fake[iH - 4] = new THStack(Form("stack_trk_len_%d_3f", iH), Form("trk_len=%d, 3 Fake", iH));
    stackLength3Fake[iH - 4]->Add(histLength3Fake[iH - 4]);
    stackLength3Fake[iH - 4]->Add(histLength3FakeNoCl[iH - 4]);
  }
}

void TrackCheckStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  initialiseRun(recoData);
  process();
}

void TrackCheckStudy::initialiseRun(o2::globaltracking::RecoContainer& recoData)
{
  mTracksROFRecords = recoData.getITSTracksROFRecords();
  mTracks = recoData.getITSTracks();
  mTracksMCLabels = recoData.getITSTracksMCLabels();
  mClusters = recoData.getITSClusters();
  mClustersMCLCont = recoData.getITSClustersMCLabels();

  LOGP(info, "** Found in {} rofs:\n\t- {} clusters with {} labels\n\t- {} tracks with {} labels",
       mTracksROFRecords.size(), mClusters.size(), mClustersMCLCont->getIndexedSize(), mTracks.size(), mTracksMCLabels.size());
  LOGP(info, "** Found {} sources from kinematic files", mKineReader->getNSources());
}

void TrackCheckStudy::process()
{
  LOGP(info, "** Filling particle table ... ");
  mParticleInfo.resize(mKineReader->getNSources()); // sources
  for (int iSource{0}; iSource < mKineReader->getNSources(); ++iSource) {
    mParticleInfo[iSource].resize(mKineReader->getNEvents(iSource)); // events
    for (int iEvent{0}; iEvent < mKineReader->getNEvents(iSource); ++iEvent) {
      mParticleInfo[iSource][iEvent].resize(mKineReader->getTracks(iSource, iEvent).size()); // tracks
      for (auto iPart{0}; iPart < mKineReader->getTracks(iEvent).size(); ++iPart) {
        auto& part = mKineReader->getTracks(iSource, iEvent)[iPart];
        mParticleInfo[iSource][iEvent][iPart].event = iEvent;
        mParticleInfo[iSource][iEvent][iPart].pdg = part.GetPdgCode();
        mParticleInfo[iSource][iEvent][iPart].pt = part.GetPt();
        mParticleInfo[iSource][iEvent][iPart].phi = part.GetPhi();
        mParticleInfo[iSource][iEvent][iPart].eta = part.GetEta();
        mParticleInfo[iSource][iEvent][iPart].vx = part.Vx();
        mParticleInfo[iSource][iEvent][iPart].vy = part.Vy();
        mParticleInfo[iSource][iEvent][iPart].vz = part.Vz();
        mParticleInfo[iSource][iEvent][iPart].isPrimary = part.isPrimary();
        mParticleInfo[iSource][iEvent][iPart].mother = part.getMotherTrackId();
        // mParticleInfo[iSource][iEvent][iPart].first = part.getFirstDaughterTrackId();
      }
    }
  }
  LOGP(info, "** Creating particle/clusters correspondance ... ");
  for (auto iSource{0}; iSource < mParticleInfo.size(); ++iSource) {
    for (auto iCluster{0}; iCluster < mClusters.size(); ++iCluster) {
      auto labs = mClustersMCLCont->getLabels(iCluster); // ideally I can have more than one label per cluster
      for (auto& lab : labs) {
        if (!lab.isValid()) {
          continue; // We want to skip channels related to noise, e.g. sID = 99: QED
        }
        int trackID, evID, srcID;
        bool fake;
        const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);
        auto& cluster = mClusters[iCluster];
        auto layer = mGeometry->getLayer(cluster.getSensorID());
        mParticleInfo[srcID][evID][trackID].clusters |= (1 << layer);
      }
    }
  }
  LOGP(info, "** Analysing tracks ... ");
  int unaccounted{0}, good{0}, fakes{0}, total{0};
  int good0 = 0, good1 = 0, good2 = 0, good3 = 0, totalsec = 0, totsectake = 0; // secondary
  int fake0 = 0, fake1 = 0, fake2 = 0, fake3 = 0;
  int totsec0 = 0, totsec1 = 0, totsec2 = 0, totsec3 = 0;
  int nlayer = 999;
  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
    auto& lab = mTracksMCLabels[iTrack];
    if (!lab.isSet() || lab.isNoise()) {
      unaccounted++;
      continue;
    }
    int trackID, evID, srcID;
    bool fake;
    const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);
    bool pass{true};

    if (srcID == 99) { // skip QED
      unaccounted++;
      continue;
    }

    mParticleInfo[srcID][evID][trackID].isReco += !fake;
    mParticleInfo[srcID][evID][trackID].isFake += fake;
    if (mTracks[iTrack].isBetter(mParticleInfo[srcID][evID][trackID].track, 1.e9)) {
      mParticleInfo[srcID][evID][trackID].storedStatus = fake;
      mParticleInfo[srcID][evID][trackID].track = mTracks[iTrack];
    }
    fakes += fake;
    good += !fake;
  }
  LOGP(info, "** Some statistics:");
  LOGP(info, "\t- Total number of tracks: {}", mTracks.size());
  LOGP(info, "\t- Total number of tracks not corresponding to particles: {} ({:.2f} %)", unaccounted, unaccounted * 100. / mTracks.size());
  LOGP(info, "\t- Total number of fakes: {} ({:.2f} %)", fakes, fakes * 100. / mTracks.size());
  LOGP(info, "\t- Total number of good: {} ({:.2f} %)", good, good * 100. / mTracks.size());

  LOGP(info, "** Filling histograms ... ");
  int evID = 0;
  int trackID = 0;
  int Ltot, Ktot, Itot;
  int LtotG, KtotG, ItotG;
  int LtotF, KtotF, ItotF;
  // Currently process only sourceID = 0, to be extended later if needed
  for (auto& evInfo : mParticleInfo[0]) {
    trackID = 0;
    for (auto& part : evInfo) {
      if ((part.clusters & 0x7f) == mMask) {
        // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
        // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters != 0x1f << 2 &&
        // part.clusters != 0x0f && part.clusters != 0x0f << 1 && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
        // continue;

        if (part.isPrimary) { // **Primary particle**
          mDenominatorPt->Fill(part.pt);
          mDenominatorEta->Fill(part.eta);
          if (part.isReco) {
            mGoodPt->Fill(part.pt);
            mGoodEta->Fill(part.eta);
            if (part.isReco > 1) {
              for (int _i{0}; _i < part.isReco - 1; ++_i) {
                mClonePt->Fill(part.pt);
                mCloneEta->Fill(part.eta);
              }
            }
          }
          if (part.isFake) {
            mFakePt->Fill(part.pt);
            mFakeEta->Fill(part.eta);
            if (part.isFake > 1) {
              for (int _i{0}; _i < part.isFake - 1; ++_i) {
                mMultiFake->Fill(part.pt);
              }
            }
          }
        }
      }

      if (!part.isPrimary) { // **Secondary particle**
        totalsec++;
        int pdgcode = mParticleInfo[0][evID][part.mother].pdg;
        float rad = sqrt(pow(part.vx, 2) + pow(part.vy, 2));

        if ((rad < rLayer0) && (part.clusters == 0x7f || part.clusters == 0x3f || part.clusters == 0x1f || part.clusters == 0x0f)) // layer 0
          nlayer = 0;
        if (rad < rLayer1 && rad > rLayer0 && (part.clusters == 0x1e || part.clusters == 0x3e || part.clusters == 0x7e)) // layer 1
          nlayer = 1;
        if (rad < rLayer2 && rad > rLayer1 && (part.clusters == 0x7c || part.clusters == 0x3c)) // layer 2
          nlayer = 2;
        if (rad < rLayer3 && rad > rLayer2 && part.clusters == 0x78) // layer 3
          nlayer = 3;
        if (nlayer == 0 || nlayer == 1 || nlayer == 2 || nlayer == 3) {
          mDenominatorSecRad->Fill(rad);
          mDenominatorSecZ->Fill(part.vz);
          totsectake++;
          mTotPts[nlayer]->Fill(part.pt);
          mTotEtas[nlayer]->Fill(part.eta);

          if (part.isReco) {
            mGoodPts[nlayer]->Fill(part.pt);
            mGoodEtas[nlayer]->Fill(part.eta);
          }
          if (part.isFake) {
            mFakePts[nlayer]->Fill(part.pt);
            mFakeEtas[nlayer]->Fill(part.eta);
          }
          if (pdgcode == 310) // k0s
          {
            mRadk->Fill(rad);
            mZk->Fill(part.vz);
            mTotPtsK[nlayer]->Fill(part.pt);
            if (part.isReco) {
              mGoodRadk->Fill(rad);
              mGoodZk->Fill(part.vz);
              mGoodPtsK[nlayer]->Fill(part.pt);
            }
            if (part.isFake) {
              mFakeRadk->Fill(rad);
              mFakeZk->Fill(part.vz);
              mFakePtsK[nlayer]->Fill(part.pt);
            }
          }
          if (pdgcode == 3122) // Lambda
          {
            mRadLam->Fill(rad);
            mZLam->Fill(part.vz);
            mTotPtsLam[nlayer]->Fill(part.pt);
            if (part.isReco) {
              mGoodRadLam->Fill(rad);
              mGoodZLam->Fill(part.vz);
              mGoodPtsLam[nlayer]->Fill(part.pt);
            }
            if (part.isFake) {
              mFakeRadLam->Fill(rad);
              mFakeZLam->Fill(part.vz);
              mFakePtsLam[nlayer]->Fill(part.pt);
            }
          }
          if (pdgcode == 1010010030) // IperT
          {
            mTotPtsIperT[nlayer]->Fill(part.pt);
            if (part.isReco) {
              mGoodPtsIperT[nlayer]->Fill(part.pt);
            }
            if (part.isFake) {
              mFakePtsIperT[nlayer]->Fill(part.pt);
            }
          }

          // Analysing fake clusters

          int nCl{0};
          for (unsigned int bit{0}; bit < sizeof(part.clusters) * 8; ++bit) {
            nCl += bool(part.clusters & (1 << bit));
          }
          if (nCl < 3) {
            continue;
          }
          auto& track = part.track;
          auto len = track.getNClusters();
          for (int iLayer{0}; iLayer < 7; ++iLayer) {
            if (track.hasHitOnLayer(iLayer)) {
              if (track.isFakeOnLayer(iLayer)) {       // Reco track has fake cluster
                if (part.clusters & (0x1 << iLayer)) { // Correct cluster exists
                  histLength[len - 4]->Fill(iLayer);
                  if (track.getNFakeClusters() == 1) {
                    histLength1Fake[len - 4]->Fill(iLayer);
                  }
                  if (track.getNFakeClusters() == 2) {
                    histLength2Fake[len - 4]->Fill(iLayer);
                  }
                  if (track.getNFakeClusters() == 3) {
                    histLength3Fake[len - 4]->Fill(iLayer);
                  }
                } else {
                  histLengthNoCl[len - 4]->Fill(iLayer);
                  if (track.getNFakeClusters() == 1) {
                    histLength1FakeNoCl[len - 4]->Fill(iLayer);
                  }
                  if (track.getNFakeClusters() == 2) {
                    histLength2FakeNoCl[len - 4]->Fill(iLayer);
                  }
                  if (track.getNFakeClusters() == 3) {
                    histLength3FakeNoCl[len - 4]->Fill(iLayer);
                  }
                }
              }
            }
          }
        }
      }
      nlayer = 999;
      trackID++;
    }

    evID++;
  }

  HistoMC.push_back(*mDenominatorSecRad);
  HistoMC.push_back(*mRadk);
  HistoMC.push_back(*mRadLam);
  HistoMC.push_back(*mDenominatorSecZ);
  HistoMC.push_back(*mZk);
  HistoMC.push_back(*mZLam);

  LOGP(info, "** Some statistics on secondary tracks:");

  LOGP(info, "\t- Total number of secondary tracks: {}", totalsec);
  LOGP(info, "\t- Total number of secondary tracks on layer O: {}, good: {}, fake: {}", totsec0, good0, fake0);
  LOGP(info, "\t- Total number of secondary tracks on layer 1: {}, good: {}, fake: {}", totsec1, good1, fake1);
  LOGP(info, "\t- Total number of secondary tracks on layer 2: {}, good: {}, fake: {}", totsec2, good2, fake2);
  LOGP(info, "\t- Total number of secondary tracks on layer 3: {}, good: {}, fake: {}", totsec3, good3, fake3);
  LOGP(info, "fraction of k = {}, fraction of lambda= {}", (*mRadk).GetEntries() / (*mDenominatorSecRad).GetEntries(), (*mRadLam).GetEntries() / (*mDenominatorSecRad).GetEntries());
  LOGP(info, "** Computing efficiencies ...");

  mEffPt = std::make_unique<TEfficiency>(*mGoodPt, *mDenominatorPt);
  mEffFakePt = std::make_unique<TEfficiency>(*mFakePt, *mDenominatorPt);
  mEffClonesPt = std::make_unique<TEfficiency>(*mClonePt, *mDenominatorPt);

  mEffEta = std::make_unique<TEfficiency>(*mGoodEta, *mDenominatorEta);
  mEffFakeEta = std::make_unique<TEfficiency>(*mFakeEta, *mDenominatorEta);
  mEffClonesEta = std::make_unique<TEfficiency>(*mCloneEta, *mDenominatorEta);

  mEffRad = std::make_unique<TEfficiency>(*mGoodRad, *mDenominatorSecRad);
  mEffFakeRad = std::make_unique<TEfficiency>(*mFakeRad, *mDenominatorSecRad);

  mEffZ = std::make_unique<TEfficiency>(*mGoodZ, *mDenominatorSecZ);
  mEffFakeZ = std::make_unique<TEfficiency>(*mFakeZ, *mDenominatorSecZ);

  mEffRadk = std::make_unique<TEfficiency>(*mGoodRadk, *mRadk);
  mEffFakeRadk = std::make_unique<TEfficiency>(*mFakeRadk, *mRadk);

  mEffZk = std::make_unique<TEfficiency>(*mGoodZk, *mZk);
  mEffFakeZk = std::make_unique<TEfficiency>(*mFakeZk, *mZk);

  mEffRadLam = std::make_unique<TEfficiency>(*mGoodRadLam, *mRadLam);
  mEffFakeRadLam = std::make_unique<TEfficiency>(*mFakeRadLam, *mRadLam);

  mEffZLam = std::make_unique<TEfficiency>(*mGoodZLam, *mZLam);
  mEffFakeZLam = std::make_unique<TEfficiency>(*mFakeZLam, *mZLam);

  for (int ii = 0; ii < 4; ii++) {
    mEffGoodPts[ii] = std::make_unique<TEfficiency>(*mGoodPts[ii], *mTotPts[ii]);
    mEffFakePts[ii] = std::make_unique<TEfficiency>(*mFakePts[ii], *mTotPts[ii]);
    mEffGoodEtas[ii] = std::make_unique<TEfficiency>(*mGoodEtas[ii], *mTotEtas[ii]);
    mEffFakeEtas[ii] = std::make_unique<TEfficiency>(*mFakeEtas[ii], *mTotEtas[ii]);

    mEffGoodPtsK[ii] = std::make_unique<TEfficiency>(*mGoodPtsK[ii], *mTotPtsK[ii]);
    mEffFakePtsK[ii] = std::make_unique<TEfficiency>(*mFakePtsK[ii], *mTotPtsK[ii]);
    mEffGoodPtsLam[ii] = std::make_unique<TEfficiency>(*mGoodPtsLam[ii], *mTotPtsLam[ii]);
    mEffFakePtsLam[ii] = std::make_unique<TEfficiency>(*mFakePtsLam[ii], *mTotPtsLam[ii]);
    mEffGoodPtsIperT[ii] = std::make_unique<TEfficiency>(*mGoodPtsIperT[ii], *mTotPtsIperT[ii]);
    mEffFakePtsIperT[ii] = std::make_unique<TEfficiency>(*mFakePtsIperT[ii], *mTotPtsIperT[ii]);
  }

  LOGP(info, "** Analysing pT resolution...");
  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
    auto& lab = mTracksMCLabels[iTrack];
    if (!lab.isSet() || lab.isNoise())
      continue;
    int trackID, evID, srcID;
    bool fake;
    const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);
    bool pass{true};
    if (srcID == 99)
      continue; // skip QED
    mPtResolution->Fill((mParticleInfo[srcID][evID][trackID].pt - mTracks[iTrack].getPt()) / mParticleInfo[srcID][evID][trackID].pt);
    mPtResolution2D->Fill(mParticleInfo[srcID][evID][trackID].pt, (mParticleInfo[srcID][evID][trackID].pt - mTracks[iTrack].getPt()) / mParticleInfo[srcID][evID][trackID].pt);
    if (!mParticleInfo[srcID][evID][trackID].isPrimary)
      mPtResolutionSec->Fill((mParticleInfo[srcID][evID][trackID].pt - mTracks[iTrack].getPt()) / mParticleInfo[srcID][evID][trackID].pt);
    mPtResolutionPrim->Fill((mParticleInfo[srcID][evID][trackID].pt - mTracks[iTrack].getPt()) / mParticleInfo[srcID][evID][trackID].pt);
  }

  for (int yy = 0; yy < 100; yy++) {
    aa[yy] = 0.;
    sigma[yy] = 0.;
    sigmaerr[yy] = 0.;
    meanPt[yy] = 0.;
  }

  for (int yy = 0; yy < 100; yy++) {
    TH1D* projh2X = mPtResolution2D->ProjectionY("projh2X", yy, yy + 1, "");
    TF1* f1 = new TF1("f1", "gaus", -0.2, 0.2);
    projh2X->Fit("f1");
    if (f1->GetParameter(2) > 0. && f1->GetParameter(2) < 1. && f1->GetParameter(1) < 1.) {
      sigma[yy] = f1->GetParameter(2);
      sigmaerr[yy] = f1->GetParError(2);
      meanPt[yy] = ((8. / 100.) * yy + (8. / 100.) * (yy + 1)) / 2;
      aa[yy] = 0.0125;
    }
  }

  LOGP(info, "** Analysing fake clusters ** ... ");
  for (auto iSource{0}; iSource < mParticleInfo.size(); ++iSource) {
    for (auto iCluster{0}; iCluster < mClusters.size(); ++iCluster) {
      auto labs = mClustersMCLCont->getLabels(iCluster); // ideally I can have more than one label per cluster
      for (auto& lab : labs) {
        if (!lab.isValid()) {
          continue; // We want to skip channels related to noise, e.g. sID = 99: QED
        }
        int trackID, evID, srcID;
        bool fake;
        const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);
        
      }
    }
  }
}

void TrackCheckStudy::setEfficiencyGraph(std::unique_ptr<TEfficiency>& eff, const char* name, const char* title, const int color, const double alpha = 1, const double linew = 2, const int markerStyle = kFullCircle, const double markersize = 1.7)
{
  eff->SetName(name);
  eff->SetTitle(title);
  eff->SetLineColor(color);
  eff->SetLineColorAlpha(color, alpha);
  eff->SetMarkerColor(color);
  eff->SetMarkerColorAlpha(color, alpha);
  eff->SetLineWidth(linew);
  eff->SetMarkerStyle(markerStyle);
  eff->SetMarkerSize(markersize);
  eff->SetDirectory(gDirectory);
}

void TrackCheckStudy::setHistoMCGraph(TH1D& histo, std::unique_ptr<TH1D>& histo2, const char* name, const char* title, const int color, const double alpha = 0.5)
{
  histo.SetName(name);
  histo2->SetName(name);
  histo.SetTitle(title);
  histo.SetFillColor(color);
  histo.SetFillColorAlpha(color, alpha);
  histo.SetDirectory(gDirectory);
}

void TrackCheckStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    mGeometry = GeometryTGeo::Instance();
    mGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void TrackCheckStudy::endOfStream(EndOfStreamContext& ec)
{
  TFile fout(mOutFileName.c_str(), "recreate");
  NormalizeHistos(HistoMC);
  // std::unique_ptr<TEfficiency> mEffPt1;
  // mEffPt1=mEffPt->Clone();

  setEfficiencyGraph(mEffPt, "Good_pt", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kAzure + 4, 0.65);
  fout.WriteTObject(mEffPt.get());

  setEfficiencyGraph(mEffFakePt, "Fake_pt", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kRed, 0.65);
  fout.WriteTObject(mEffFakePt.get());

  setEfficiencyGraph(mEffPt, "Tot Prim G Tracks", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kBlack, 1);
  setEfficiencyGraph(mEffFakePt, "Tot Prim F Tracks", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kBlack, 1);

  setEfficiencyGraph(mEffClonesPt, "Clone_pt", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kGreen + 2, 0.65);
  fout.WriteTObject(mEffClonesPt.get());

  setEfficiencyGraph(mEffEta, "Good_eta", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kAzure + 4, 0.65);
  fout.WriteTObject(mEffEta.get());

  setEfficiencyGraph(mEffFakeEta, "Fake_eta", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kRed + 1, 0.65);
  fout.WriteTObject(mEffFakeEta.get());

  setEfficiencyGraph(mEffClonesEta, "Clone_eta", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kGreen + 2, 0.65);
  fout.WriteTObject(mEffClonesEta.get());

  setEfficiencyGraph(mEffRad, "Good_Rad", ";Radius [cm];efficiency secondary particle", kBlue, 1);
  fout.WriteTObject(mEffRad.get());

  setEfficiencyGraph(mEffFakeRad, "Fake_Rad", ";Radius [cm];efficiency secondary particle", kOrange + 7, 1);
  fout.WriteTObject(mEffFakeRad.get());

  setEfficiencyGraph(mEffRadk, "Good_Radk", ";Radius [cm];efficiency secondary particle", kBlue, 1);
  fout.WriteTObject(mEffRadk.get());

  setEfficiencyGraph(mEffFakeRadk, "Fake_Radk", ";Radius [cm];efficiency secondary particle", kAzure + 10, 1);
  fout.WriteTObject(mEffFakeRadk.get());

  setEfficiencyGraph(mEffRadLam, "Good_RadLam", ";Radius [cm];efficiency secondary particle", kRed + 2, 1);
  fout.WriteTObject(mEffRadLam.get());

  setEfficiencyGraph(mEffFakeRadLam, "Fake_RadLam", ";Radius [cm];efficiency secondary particle", kMagenta - 9, 1);
  fout.WriteTObject(mEffFakeRadLam.get());

  setEfficiencyGraph(mEffZ, "Good_Z", ";z secondary vertex [cm];efficiency secondary particle", kTeal + 2, 1);
  fout.WriteTObject(mEffZ.get());

  setEfficiencyGraph(mEffFakeZ, "Fake_Z", ";z secondary vertex [cm];efficiency secondary particle", kMagenta - 4, 1);
  fout.WriteTObject(mEffFakeZ.get());

  setEfficiencyGraph(mEffZk, "Good_Zk", ";z of sv  [cm];efficiency secondary particle", kBlue);
  fout.WriteTObject(mEffZk.get());

  setEfficiencyGraph(mEffFakeZk, "Fake_Zk", ";z of sv  [cm];efficiency secondary particle", kAzure + 10);
  fout.WriteTObject(mEffFakeZk.get());

  setEfficiencyGraph(mEffZLam, "Good_ZLam", ";z of sv  [cm];efficiency secondary particle", kRed + 2);
  fout.WriteTObject(mEffZLam.get());

  setEfficiencyGraph(mEffFakeZLam, "Fake_ZLam", ";z of sv  [cm];efficiency secondary particle", kMagenta - 9);
  fout.WriteTObject(mEffFakeZLam.get());

  for (int aa = 0; aa < 4; aa++) {
    setEfficiencyGraph(mEffGoodPts[aa], Form("EffPtGoodl%d", aa), Form("Tot Sec G Tracks, L%d"
                                                                       ";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ",
                                                                       aa),
                       kGray);
    setEfficiencyGraph(mEffFakePts[aa], Form("EffPtFakel%d", aa), Form("Tot Sec F Tracks, L%d"
                                                                       ";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ",
                                                                       aa),
                       kGray);
    setEfficiencyGraph(mEffGoodEtas[aa], Form("EffEtaGoodl%d", aa), Form("Tot Sec G Tracks, L%d"
                                                                         ";eta ;efficiency secondary particle ",
                                                                         aa),
                       kAzure + 4);
    setEfficiencyGraph(mEffFakeEtas[aa], Form("EffEtaFakel%d", aa), Form("Tot Sec F Tracks, L%d"
                                                                         ";eta ;efficiency secondary particle ",
                                                                         aa),
                       kAzure + 4);
    setEfficiencyGraph(mEffGoodPtsK[aa], Form("EffPtGoodKl%d", aa), Form("k^{0}_{s} Sec G Tracks, L%d ;#it{p}_{T} (GeV/#it{c});efficiency secondary particle ", aa), kBlue);
    setEfficiencyGraph(mEffFakePtsK[aa], Form("EffPtFakeKl%d", aa), Form("k^{0}_{s} Sec F Tracks, L%d ;#it{p}_{T} (GeV/#it{c});efficiency secondary particle ", aa), kBlue);
    setEfficiencyGraph(mEffGoodPtsLam[aa], Form("EffPtGoodLaml%d", aa), Form("#Lambda Sec G Tracks, L%d ;#it{p}_{T} (GeV/#it{c});efficiency secondary particle ", aa), kRed);
    setEfficiencyGraph(mEffFakePtsLam[aa], Form("EffPtFakeLaml%d", aa), Form("#Lambda Sec F Tracks, L%d ;#it{p}_{T} (GeV/#it{c});efficiency secondary particle ", aa), kRed);
    setEfficiencyGraph(mEffGoodPtsIperT[aa], Form("EffPtGoodIperTl%d", aa), Form("_{#Lambda}^{3}H Sec G Tracks, L%d ;#it{p}_{T} (GeV/#it{c});efficiency secondary particle ", aa), kGreen + 1);
    setEfficiencyGraph(mEffFakePtsIperT[aa], Form("EffPtFakeIperTl%d", aa), Form("_{#Lambda}^{3}H Sec F Tracks, L%d ;#it{p}_{T} (GeV/#it{c});efficiency secondary particle ", aa), kGreen + 1);
    fout.WriteTObject(mEffGoodPts[aa].get());
    fout.WriteTObject(mEffFakePts[aa].get());
    fout.WriteTObject(mEffGoodEtas[aa].get());
    fout.WriteTObject(mEffFakeEtas[aa].get());
    fout.WriteTObject(mEffGoodPtsK[aa].get());
    fout.WriteTObject(mEffGoodPtsLam[aa].get());
    fout.WriteTObject(mEffGoodPtsIperT[aa].get());
    fout.WriteTObject(mEffFakePtsK[aa].get());
    fout.WriteTObject(mEffFakePtsLam[aa].get());
    fout.WriteTObject(mEffFakePtsIperT[aa].get());
  }
  setHistoMCGraph(HistoMC[0], mDenominatorSecRad, "Decay_Radius_MC", ";Decay Radius ;Entries", kGray, 0.4);
  fout.WriteTObject(mDenominatorSecRad.get());

  setHistoMCGraph(HistoMC[3], mDenominatorSecZ, "Zsv_MC", ";z of secondary vertex ;Entries", kGray, 0.4);
  fout.WriteTObject(mDenominatorSecZ.get());

  setHistoMCGraph(HistoMC[1], mRadk, "Decay_Radius_MC_k", ";Decay Radius ;Entries", kBlue, 0.2);
  fout.WriteTObject(mRadk.get());

  setHistoMCGraph(HistoMC[4], mZk, "Zsv_MC_k", ";Zsv ;Entries", kBlue, 0.2);
  fout.WriteTObject(mZk.get());

  setHistoMCGraph(HistoMC[2], mRadLam, "Decay_Radius_MC_Lam", ";Decay Radius ;Entries", kRed, 0.2);
  fout.WriteTObject(mRadLam.get());

  setHistoMCGraph(HistoMC[5], mZLam, "Zsv_MC_Lam", ";Zsv ;Entries", kRed, 0.2);
  fout.WriteTObject(mZLam.get());

  // Paint the histograms
  // todo:  delegate to a dedicated helper
  gStyle->SetTitleSize(0.035, "xy");
  gStyle->SetLabelSize(0.035, "xy");
  gStyle->SetPadRightMargin(0.035);
  gStyle->SetPadTopMargin(0.035);
  gStyle->SetPadLeftMargin(0.19);
  gStyle->SetPadBottomMargin(0.17);
  gStyle->SetTitleOffset(1.4, "x");
  gStyle->SetTitleOffset(1.1, "y");
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetGridStyle(3);
  gStyle->SetGridWidth(1);

  mCanvasPt = std::make_unique<TCanvas>("cPt", "cPt", 1600, 1200);
  mCanvasPt->cd();
  mCanvasPt->SetLogx();
  mCanvasPt->SetGrid();
  mEffPt->Draw("pz");
  mEffFakePt->Draw("pz same");
  mEffClonesPt->Draw("pz same");
  mLegendPt = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendPt->SetHeader(Form("%zu events PbPb min bias", mKineReader->getNEvents(0)), "C");
  mLegendPt->AddEntry("Good_pt", "good (100% cluster purity)", "lep");
  mLegendPt->AddEntry("Fake_pt", "fake", "lep");
  mLegendPt->AddEntry("Clone_pt", "clone", "lep");
  mLegendPt->Draw();
  mCanvasPt->SaveAs("eff_pt.png");

  mCanvasEta = std::make_unique<TCanvas>("cEta", "cEta", 1600, 1200);
  mCanvasEta->cd();
  mCanvasEta->SetGrid();
  mEffEta->Draw("pz");
  mEffFakeEta->Draw("pz same");
  mEffClonesEta->Draw("pz same");
  mLegendEta = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendEta->SetHeader(Form("%zu events PbPb min bias", mKineReader->getNEvents(0)), "C");
  mLegendEta->AddEntry("Good_eta", "good (100% cluster purity)", "lep");
  mLegendEta->AddEntry("Fake_eta", "fake", "lep");
  mLegendEta->AddEntry("Clone_eta", "clone", "lep");
  mLegendEta->Draw();
  mCanvasEta->SaveAs("eff_eta.png");

  mCanvasRad = std::make_unique<TCanvas>("cRad", "cRad", 1600, 1200);
  mCanvasRad->cd();
  mCanvasRad->SetGrid();
  mEffRad->Draw("pz");
  mEffFakeRad->Draw("pz same");
  HistoMC[0].Draw("hist same");
  mCanvasRad->SetLogy();
  mLegendRad = std::make_unique<TLegend>(0.8, 0.4, 0.95, 0.6);
  mLegendRad->SetHeader(Form("%zu events PP ", mKineReader->getNEvents(0)), "C");
  mLegendRad->AddEntry("Good_Rad", "good", "lep");
  mLegendRad->AddEntry("Fake_Rad", "fake", "lep");
  mLegendRad->AddEntry("Decay_Radius_MC", "MC", "f");
  mLegendRad->Draw();
  mCanvasRad->SaveAs("eff_rad_sec_MC.png");

  mCanvasZ = std::make_unique<TCanvas>("cZ", "cZ", 1600, 1200);
  mCanvasZ->cd();
  mCanvasZ->SetGrid();
  mCanvasZ->SetLogy();
  mEffZ->Draw("pz");
  mEffFakeZ->Draw("pz same");
  HistoMC[3].Draw(" histsame");
  mLegendZ = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendZ->SetHeader(Form("%zu events PP", mKineReader->getNEvents(0)), "C");
  mLegendZ->AddEntry("Good_Z", "good", "lep");
  mLegendZ->AddEntry("Fake_Z", "fake", "lep");
  mLegendZ->AddEntry("Zsv_MC", "MC", "f");

  mLegendZ->Draw();
  mCanvasZ->SaveAs("eff_Z_sec_MC.png");

  mCanvasRadD = std::make_unique<TCanvas>("cRadD", "cRadD", 1600, 1200);
  mCanvasRadD->cd();
  mCanvasRadD->SetGrid();
  mEffRadk->Draw("pz");
  mEffFakeRadk->Draw("pz same");
  HistoMC[1].Draw(" hist same");
  mEffRadLam->Draw("pz same");
  mEffFakeRadLam->Draw("pz same");
  HistoMC[2].Draw(" hist same");
  mCanvasRadD->SetLogy();
  mLegendRadD = std::make_unique<TLegend>(0.8, 0.64, 0.95, 0.8);
  mLegendRadD->SetHeader(Form("%zu events PP ", mKineReader->getNEvents(0)), "C");
  mLegendRadD->AddEntry("Good_Radk", " k^{0}_{s} good", "lep");
  mLegendRadD->AddEntry("Fake_Radk", "k^{0}_{s} fake", "lep");
  mLegendRadD->AddEntry("Decay_Radius_MC_k", "k^{0}_{s} MC", "f");
  mLegendRadD->AddEntry("Good_RadLam", " #Lambda good", "lep");
  mLegendRadD->AddEntry("Fake_RadLam", "#Lambda fake", "lep");
  mLegendRadD->AddEntry("Decay_Radius_MC_Lam", "#Lambda MC", "f");

  mLegendRadD->Draw();
  mCanvasRadD->SaveAs("eff_RadD_sec_MC.png");

  mCanvasZD = std::make_unique<TCanvas>("cZd", "cZd", 1600, 1200);
  mCanvasZD->cd();
  mCanvasZD->SetGrid();
  mEffZk->Draw("pz");
  mEffFakeZk->Draw("pz same");
  HistoMC[4].Draw("same hist");
  mEffZLam->Draw("pz same");
  mEffFakeZLam->Draw("pz same");
  HistoMC[5].Draw("same hist");
  mLegendZD = std::make_unique<TLegend>(0.19, 0.5, 0.30, 0.7);
  mLegendZD->SetHeader(Form("%zu events PP", mKineReader->getNEvents(0)), "C");
  mLegendZD->AddEntry("Good_Zk", " k^{0}_{s} good", "lep");
  mLegendZD->AddEntry("Fake_Zk", "k^{0}_{s} fake", "lep");
  mLegendZD->AddEntry("Zsv_MC_k", "k^{0}_{s} MC", "f");
  mLegendZD->AddEntry("Good_ZLam", "#Lambda good", "lep");
  mLegendZD->AddEntry("Fake_ZLam", "#Lambda fake", "lep");
  mLegendZD->AddEntry("Zsv_MC_Lam", "#Lambda MC", "f");

  mLegendZD->Draw();
  mCanvasZD->SaveAs("eff_ZD_sec_MC.png");

  mPtResolution->SetName("#it{p}_{T} resolution");
  mPtResolution->SetTitle(";#Delta p_{T}/p_{T_{MC}} ;Entries");
  mPtResolution->SetFillColor(kAzure + 4);
  mPtResolutionPrim->SetFillColor(kRed);
  mPtResolutionSec->SetFillColor(kOrange);
  mPtResolutionPrim->SetTitle(";#Delta p_{T}/p_{T_{MC}} ;Entries");
  mPtResolutionSec->SetTitle(";#Delta #it{p}_{T}/#it{p}_{T_{MC}} ;Entries");
  mPtResolution2D->SetTitle(";#it{p}_{T_{MC}} [GeV];#Delta #it{p}_{T}/#it{p}_{T_{MC}}");

  fout.WriteTObject(mPtResolution.get());
  fout.WriteTObject(mPtResolutionPrim.get());
  fout.WriteTObject(mPtResolutionSec.get());
  fout.WriteTObject(mPtResolution2D.get());

  mCanvasPtRes = std::make_unique<TCanvas>("cPtr", "cPtr", 1600, 1200);
  mCanvasPtRes->cd();
  mPtResolution->Draw("HIST");
  mLegendPtRes = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendPtRes->SetHeader(Form("%zu events PP min bias", mKineReader->getNEvents(0)), "C");
  mLegendPtRes->AddEntry("mPtResolution", "All events", "lep");
  mLegendPtRes->Draw();
  mCanvasPtRes->SaveAs("ptRes.png");

  mCanvasPtRes2 = std::make_unique<TCanvas>("cPtr2", "cPtr2", 1600, 1200);
  mCanvasPtRes2->cd();
  mPtResolution2D->Draw();
  mCanvasPtRes2->SaveAs("ptRes2.png");

  mCanvasPtRes3 = std::make_unique<TCanvas>("cPtr3", "cPtr3", 1600, 1200);
  mCanvasPtRes3->cd();

  TGraphErrors* g1 = new TGraphErrors(100, meanPt, sigma, aa, sigmaerr);
  g1->SetMarkerStyle(8);
  g1->SetMarkerColor(kGreen);
  g1->GetXaxis()->SetTitle(" #it{p}_{T} [GeV]");
  g1->GetYaxis()->SetTitle("#sigma #Delta #it{p}_{T}/#it{p}_{T_{MC}}");
  g1->GetYaxis()->SetLimits(0, 1);
  g1->GetXaxis()->SetLimits(0, 10.);
  g1->Draw("AP");
  g1->GetYaxis()->SetRangeUser(0, 1);
  g1->GetXaxis()->SetRangeUser(0, 10.);
  mCanvasPtRes3->SaveAs("ptRes3.png");

  mCanvasPtRes4 = std::make_unique<TCanvas>("cPt4", "cPt4", 1600, 1200);
  mCanvasPtRes4->cd();
  mPtResolutionPrim->SetName("mPtResolutionPrim");
  mPtResolutionSec->SetName("mPtResolutionSec");
  mPtResolutionPrim->Draw("same hist");
  mPtResolutionSec->Draw("same hist");
  mLegendPtRes2 = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);

  mLegendPtRes2->SetHeader(Form("%zu events PP", mKineReader->getNEvents(0)), "C");
  mLegendPtRes2->AddEntry("mPtResolutionPrim", "Primary events", "f");
  mLegendPtRes2->AddEntry("mPtResolutionSec", "Secondary events", "f");
  mLegendPtRes2->Draw("same");
  mLegendPtRes2->SaveAs("ptRes4.png");

  auto canvas = new TCanvas("fc_canvas", "Fake clusters", 1600, 1000);
  canvas->Divide(4, 2);
  for (int iH{0}; iH < 4; ++iH) {
    canvas->cd(iH + 1);
    stackLength[iH]->Draw();
    gPad->BuildLegend();
  }
  for (int iH{0}; iH < 4; ++iH) {
    canvas->cd(iH + 5);
    stackLength1Fake[iH]->Draw();
    gPad->BuildLegend();
  }

  canvas->SaveAs("fakeClusters2.png", "recreate");

  auto canvas2 = new TCanvas("fc_canvas2", "Fake clusters", 1600, 1000);
  canvas2->Divide(4, 2);

  for (int iH{0}; iH < 4; ++iH) {
    canvas2->cd(iH + 1);
    stackLength2Fake[iH]->Draw();
    gPad->BuildLegend();
  }
  for (int iH{0}; iH < 4; ++iH) {
    canvas2->cd(iH + 5);
    stackLength3Fake[iH]->Draw();
    gPad->BuildLegend();
  }
  canvas2->SaveAs("fakeClusters3.png", "recreate");

  auto canvasPtfake = new TCanvas("canvasPtfake", "Fake clusters", 1600, 1000);
  canvasPtfake->Divide(2, 2);

  for (int iH{0}; iH < 4; ++iH) {
    canvasPtfake->cd(iH + 1);
    mEffFakePtsK[iH]->Draw();
    mEffFakePtsLam[iH]->Draw("same");
    mEffFakePtsIperT[iH]->Draw("same");
    mEffFakePt->Draw("same");
    mEffFakePts[iH]->Draw("same");
    gPad->BuildLegend();
    gPad->SetGrid();
    gPad->SetTitle(Form("#it{p}_{T}, Fake Tracks, layer %d", iH));
    gPad->SetName(Form("#it{p}_{T}, Fake Tracks, layer %d", iH));
  }
  canvasPtfake->SaveAs("PtforPart1.png", "recreate");

  fout.cd();
  mCanvasPt->Write();
  mCanvasEta->Write();
  mCanvasPtRes->Write();
  mCanvasPtRes2->Write();
  mCanvasPtRes3->Write();
  mCanvasPtRes4->Write();
  mCanvasRad->Write();
  mCanvasZ->Write();
  mCanvasRadD->Write();
  mCanvasZD->Write();
  canvas->Write();
  canvas2->Write();
  canvasPtfake->Write();
  fout.Close();
}

void TrackCheckStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
}

DataProcessorSpec getTrackCheckStudy(mask_t srcTracksMask, mask_t srcClustersMask, bool useMC, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, useMC);
  dataRequest->requestClusters(srcClustersMask, useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    "its-study-check-tracks",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackCheckStudy>(dataRequest, srcTracksMask, useMC, kineReader, ggRequest)},
    Options{}};
}

} // namespace study
} // namespace its
} // namespace o2