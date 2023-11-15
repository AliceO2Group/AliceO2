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
#include <TH1I.h>
#include <TH1.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TEfficiency.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TObjArray.h>
#include <THStack.h>
#include <TString.h>

namespace o2::its::study
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
    bool isPrimary = false;
    unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
    const char* prodProcessName;
    int prodProcess;
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
  void setEfficiencyGraph(std::unique_ptr<TEfficiency>&, const char*, const char*, const int, const double, const double, const int, const double);

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
  gsl::span<const int> mInputITSidxs;
  const o2::dataformats::MCLabelContainer* mClustersMCLCont;

  // Data
  GTrackID::mask_t mTracksSrc{};
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<std::vector<std::vector<ParticleInfo>>> mParticleInfo; // src/event/track
  unsigned short mMask = 0x7f;

  // Utils
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;

  // Histos
  std::unique_ptr<TH1D> mGoodPt;
  std::unique_ptr<TH1D> mGoodEta;
  std::unique_ptr<TH1D> mGoodPtSec;
  std::unique_ptr<TH1D> mGoodEtaSec;
  std::unique_ptr<TH1D> mGoodChi2;
  std::unique_ptr<TH1D> mFakePt;
  std::unique_ptr<TH1D> mFakeEta;
  std::unique_ptr<TH1D> mFakePtSec;
  std::unique_ptr<TH1D> mFakeEtaSec;
  std::unique_ptr<TH1D> mMultiFake;
  std::unique_ptr<TH1D> mFakeChi2;
  std::unique_ptr<TH1D> mClonePt;
  std::unique_ptr<TH1D> mCloneEta;

  std::unique_ptr<TH1D> mDenominatorPt;
  std::unique_ptr<TH1D> mDenominatorEta;
  std::unique_ptr<TH1D> mDenominatorPtSec;
  std::unique_ptr<TH1D> mDenominatorEtaSec;

  std::unique_ptr<TH2D> processvsZ; // TH2D with production process
  std::unique_ptr<TH2D> processvsRad;
  std::unique_ptr<TH2D> processvsRadOther;
  std::unique_ptr<TH2D> processvsRadNotTracked;
  std::unique_ptr<TH2D> processvsEtaNotTracked;

  std::unique_ptr<TEfficiency> mEffPt; // Eff vs Pt primary
  std::unique_ptr<TEfficiency> mEffFakePt;
  std::unique_ptr<TEfficiency> mEffClonesPt;
  std::unique_ptr<TEfficiency> mEffEta; // Eff vs Eta primary
  std::unique_ptr<TEfficiency> mEffFakeEta;
  std::unique_ptr<TEfficiency> mEffClonesEta;

  std::unique_ptr<TEfficiency> mEffPtSec; // Eff vs Pt secondary
  std::unique_ptr<TEfficiency> mEffFakePtSec;
  std::unique_ptr<TEfficiency> mEffEtaSec; // Eff vs Eta secondary
  std::unique_ptr<TEfficiency> mEffFakeEtaSec;

  std::unique_ptr<TH1D> mPtResolution; // Pt resolution for both primary and secondary
  std::unique_ptr<TH2D> mPtResolution2D;
  std::unique_ptr<TH1D> mPtResolutionSec;
  std::unique_ptr<TH1D> mPtResolutionPrim;
  std::unique_ptr<TGraphErrors> g1;

  const char* ParticleName[7] = {"e^{-/+}", "#pi^{-/+}", "p", "^{2}H", "^{3}He", "_{#Lambda}^{3}H", "k^{+/-}"};
  const int PdgcodeClusterFake[7] = {11, 211, 2212, 1000010020, 100002030, 1010010030, 321};
  const char* name[3] = {"_{#Lambda}^{3}H", "#Lambda", "k^{0}_{s}"};
  const char* particleToanalize[4] = {"IperT", "Lambda", "k0s", "Tot"}; // [3]=Total of secondary particle
  const int PDG[3] = {1010010030, 3122, 310};
  const char* ProcessName[50];
  int colorArr[4] = {kGreen, kRed, kBlue, kOrange};

  std::vector<std::vector<TH1I*>> histLength, histLength1Fake, histLength2Fake, histLength3Fake, histLengthNoCl, histLength1FakeNoCl, histLength2FakeNoCl, histLength3FakeNoCl; // FakeCluster Study
  std::vector<THStack*> stackLength, stackLength1Fake, stackLength2Fake, stackLength3Fake;
  std::vector<TLegend*> legends, legends1Fake, legends2Fake, legends3Fake;
  std::vector<std::unique_ptr<TH2D>> mClusterFake;
  std::vector<std::vector<std::unique_ptr<TH1D>>> mGoodPts, mFakePts, mTotPts, mGoodEtas, mTotEtas, mFakeEtas;
  std::vector<std::vector<std::unique_ptr<TEfficiency>>> mEffGoodPts, mEffFakePts, mEffGoodEtas, mEffFakeEtas;
  std::vector<std::unique_ptr<TH1D>> mGoodRad, mFakeRad, mTotRad, mGoodZ, mFakeZ, mTotZ;
  std::vector<std::unique_ptr<TEfficiency>> mEffGoodRad, mEffFakeRad, mEffGoodZ, mEffFakeZ;
  //  Canvas & decorations
  std::unique_ptr<TCanvas> mCanvasPt;
  std::unique_ptr<TCanvas> mCanvasPtSec;
  std::unique_ptr<TCanvas> mCanvasPt2;
  std::unique_ptr<TCanvas> mCanvasPt2fake;
  std::unique_ptr<TCanvas> mCanvasEta;
  std::unique_ptr<TCanvas> mCanvasEtaSec;
  std::unique_ptr<TCanvas> mCanvasRad;
  std::unique_ptr<TCanvas> mCanvasZ;
  std::unique_ptr<TCanvas> mCanvasRadD;
  std::unique_ptr<TCanvas> mCanvasZD;
  std::unique_ptr<TCanvas> mCanvasPtRes;
  std::unique_ptr<TCanvas> mCanvasPtRes2;
  std::unique_ptr<TCanvas> mCanvasPtRes3;
  std::unique_ptr<TCanvas> mCanvasPtRes4;
  std::unique_ptr<TLegend> mLegendPt;
  std::unique_ptr<TLegend> mLegendEta;
  std::unique_ptr<TLegend> mLegendPtSec;
  std::unique_ptr<TLegend> mLegendEtaSec;
  std::unique_ptr<TLegend> mLegendPtRes;
  std::unique_ptr<TLegend> mLegendPtRes2;
  std::unique_ptr<TLegend> mLegendZ;
  std::unique_ptr<TLegend> mLegendRad;
  std::unique_ptr<TLegend> mLegendZD;
  std::unique_ptr<TLegend> mLegendRadD;
  std::vector<TH1D> Histo;

  float rLayer0 = 2.34; // middle radius
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
  for (int yy = 0; yy < 50; yy++) {
    ProcessName[yy] = " ";
  }
  mGoodPt = std::make_unique<TH1D>("goodPt", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
  mGoodEta = std::make_unique<TH1D>("goodEta", ";#eta;Number of tracks", 60, -3, 3);
  mGoodPtSec = std::make_unique<TH1D>("goodPtSec", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
  mGoodEtaSec = std::make_unique<TH1D>("goodEtaSec", ";#eta;Number of tracks", 60, -3, 3);
  mGoodChi2 = std::make_unique<TH1D>("goodChi2", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", 200, 0, 100);

  mFakePt = std::make_unique<TH1D>("fakePt", ";#it{p}_{T} (GeV/#it{c});Fak", pars.effHistBins, xbins.data());
  mFakeEta = std::make_unique<TH1D>("fakeEta", ";#eta;Number of tracks", 60, -3, 3);
  mFakePtSec = std::make_unique<TH1D>("fakePtSec", ";#it{p}_{T} (GeV/#it{c});Fak", pars.effHistBins, xbins.data());
  mFakeEtaSec = std::make_unique<TH1D>("fakeEtaSec", ";#eta;Number of tracks", 60, -3, 3);
  mFakeChi2 = std::make_unique<TH1D>("fakeChi2", ";#it{p}_{T} (GeV/#it{c});Fak", 200, 0, 100);

  mMultiFake = std::make_unique<TH1D>("multiFake", ";#it{p}_{T} (GeV/#it{c});Fak", pars.effHistBins, xbins.data());

  mClonePt = std::make_unique<TH1D>("clonePt", ";#it{p}_{T} (GeV/#it{c});Clone", pars.effHistBins, xbins.data());
  mCloneEta = std::make_unique<TH1D>("cloneEta", ";#eta;Number of tracks", 60, -3, 3);

  mDenominatorPt = std::make_unique<TH1D>("denominatorPt", ";#it{p}_{T} (GeV/#it{c});Den", pars.effHistBins, xbins.data());
  mDenominatorEta = std::make_unique<TH1D>("denominatorEta", ";#eta;Number of tracks", 60, -3, 3);
  mDenominatorPtSec = std::make_unique<TH1D>("denominatorPtSec", ";#it{p}_{T} (GeV/#it{c});Den", pars.effHistBins, xbins.data());
  mDenominatorEtaSec = std::make_unique<TH1D>("denominatorEtaSec", ";#eta;Number of tracks", 60, -3, 3);

  processvsZ = std::make_unique<TH2D>("Process", ";z_{SV} [cm]; production process", 100, -50, 50., 50, 0, 50);
  processvsRad = std::make_unique<TH2D>("ProcessR", ";decay radius [cm]; production process", 100, 0, 25., 50, 0, 50);
  processvsRadOther = std::make_unique<TH2D>("ProcessRO", ";decay radius [cm]; production process", 200, 0, 25., 50, 0, 50);
  processvsRadNotTracked = std::make_unique<TH2D>("ProcessRNoT", ";decay radius [cm]; production process", 200, 0, 25., 50, 0, 50);
  processvsEtaNotTracked = std::make_unique<TH2D>("ProcessENoT", ";#eta; production process", 60, -3, 3, 50, 0, 50);

  mGoodPts.resize(4);
  mFakePts.resize(4);
  mTotPts.resize(4);
  mGoodEtas.resize(4);
  mFakeEtas.resize(4);
  mTotEtas.resize(4);
  mGoodRad.resize(4);
  mFakeRad.resize(4);
  mTotRad.resize(4);
  mGoodZ.resize(4);
  mFakeZ.resize(4);
  mTotZ.resize(4);
  mClusterFake.resize(4);
  for (int i = 0; i < 4; i++) {
    mGoodPts[i].resize(4);
    mFakePts[i].resize(4);
    mTotPts[i].resize(4);
    mGoodEtas[i].resize(4);
    mFakeEtas[i].resize(4);
    mTotEtas[i].resize(4);
  }
  for (int ii = 0; ii < 4; ii++) {

    mGoodRad[ii] = std::make_unique<TH1D>(Form("goodRad_%s", particleToanalize[ii]), ";z_{SV} [cm];Number of tracks", 100, 0., 20.);
    mFakeRad[ii] = std::make_unique<TH1D>(Form("FakeRad_%s", particleToanalize[ii]), ";#eta;Number of tracks", 100, 0., 20.);
    mTotRad[ii] = std::make_unique<TH1D>(Form("TotRad_%s", particleToanalize[ii]), ";#eta;Number of tracks", 100, 0., 20.);

    mGoodZ[ii] = std::make_unique<TH1D>(Form("goodZ_%s", particleToanalize[ii]), ";z_{SV} [cm];Number of tracks", 100, -50., 50.);
    mFakeZ[ii] = std::make_unique<TH1D>(Form("FakeZ_%s", particleToanalize[ii]), ";z_{SV} [cm];Number of tracks", 100, -50., 50.);
    mTotZ[ii] = std::make_unique<TH1D>(Form("TotZ_%s", particleToanalize[ii]), ";z_{SV} [cm];Number of tracks", 100, -50., 50.);
    mClusterFake[ii] = std::make_unique<TH2D>(Form("Clusters_fake_%s", ParticleName[ii]), ";particle generating fake cluster; production process", 7, 0., 7., 50, 0, 50);

    mGoodRad[ii]->Sumw2();
    mFakeRad[ii]->Sumw2();
    mTotRad[ii]->Sumw2();
    mGoodZ[ii]->Sumw2();
    mFakeZ[ii]->Sumw2();
    mTotZ[ii]->Sumw2();

    for (int yy = 0; yy < 4; yy++) { // divided by layer
      mGoodPts[ii][yy] = std::make_unique<TH1D>(Form("goodPts_%s_%d", particleToanalize[ii], yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
      mFakePts[ii][yy] = std::make_unique<TH1D>(Form("FakePts_%s_%d", particleToanalize[ii], yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
      mTotPts[ii][yy] = std::make_unique<TH1D>(Form("TotPts_%s_%d", particleToanalize[ii], yy), ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());

      mGoodEtas[ii][yy] = std::make_unique<TH1D>(Form("goodEtas_%s_%d", particleToanalize[ii], yy), ";#eta;Number of tracks", 60, -3, 3);
      mFakeEtas[ii][yy] = std::make_unique<TH1D>(Form("FakeEtas_%s_%d", particleToanalize[ii], yy), ";#eta;Number of tracks", 60, -3, 3);
      mTotEtas[ii][yy] = std::make_unique<TH1D>(Form("TotEtas_%s_%d", particleToanalize[ii], yy), ";#eta;Number of tracks", 60, -3, 3);

      mGoodPts[ii][yy]->Sumw2();
      mFakePts[ii][yy]->Sumw2();
      mTotPts[ii][yy]->Sumw2();
      mGoodEtas[ii][yy]->Sumw2();
      mFakeEtas[ii][yy]->Sumw2();
      mTotEtas[ii][yy]->Sumw2();
    }
  }

  mPtResolution = std::make_unique<TH1D>("PtResolution", ";#it{p}_{T} ;Den", 100, -1, 1);
  mPtResolutionSec = std::make_unique<TH1D>("PtResolutionSec", ";#it{p}_{T} ;Den", 100, -1, 1);
  mPtResolutionPrim = std::make_unique<TH1D>("PtResolutionPrim", ";#it{p}_{T} ;Den", 100, -1, 1);
  mPtResolution2D = std::make_unique<TH2D>("#it{p}_{T} Resolution vs #it{p}_{T}", ";#it{p}_{T} (GeV/#it{c});#Delta p_{T}/p_{T_{MC}", 100, 0, 10, 100, -1, 1);

  mPtResolution->Sumw2();
  mPtResolutionSec->Sumw2();
  mPtResolutionPrim->Sumw2();

  mGoodPt->Sumw2();
  mGoodEta->Sumw2();
  mGoodPtSec->Sumw2();
  mGoodEtaSec->Sumw2();

  mFakePt->Sumw2();
  mFakePtSec->Sumw2();
  mFakeEta->Sumw2();
  mMultiFake->Sumw2();
  mClonePt->Sumw2();
  mDenominatorPt->Sumw2();

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
  for (int yy = 0; yy < 4; yy++) {
    histLength[yy].resize(3);
    histLength1Fake[yy].resize(3);
    histLength2Fake[yy].resize(3);
    histLength3Fake[yy].resize(3);
    histLengthNoCl[yy].resize(3);
    histLength1FakeNoCl[yy].resize(3);
    histLength2FakeNoCl[yy].resize(3);
    histLength3FakeNoCl[yy].resize(3);
  }
  legends.resize(4);
  legends1Fake.resize(4);
  legends2Fake.resize(4);
  legends3Fake.resize(4);

  for (int iH{4}; iH < 8; ++iH) {
    // check distributions on layers of fake clusters for tracks of different lengths.
    // Different histograms if the correct cluster exist or not
    for (int jj = 0; jj < 3; jj++) {
      histLength[iH - 4][jj] = new TH1I(Form("trk_len_%d_%s", iH, name[jj]), Form("#exists cluster %s", name[jj]), 7, -.5, 6.5);
      histLength[iH - 4][jj]->SetFillColor(colorArr[jj] - 9);
      histLength[iH - 4][jj]->SetLineColor(colorArr[jj] - 9);
      histLengthNoCl[iH - 4][jj] = new TH1I(Form("trk_len_%d_nocl_%s", iH, name[jj]), Form("#slash{#exists} cluster %s", name[jj]), 7, -.5, 6.5);
      histLengthNoCl[iH - 4][jj]->SetFillColor(colorArr[jj] + 1);
      histLengthNoCl[iH - 4][jj]->SetLineColor(colorArr[jj] + 1);
      if (jj == 0) {
        stackLength[iH - 4] = new THStack(Form("stack_trk_len_%d", iH), Form("trk_len=%d", iH));
      }
      stackLength[iH - 4]->Add(histLength[iH - 4][jj]);
      stackLength[iH - 4]->Add(histLengthNoCl[iH - 4][jj]);

      histLength1Fake[iH - 4][jj] = new TH1I(Form("trk_len_%d_1f_%s", iH, name[jj]), Form("#exists cluster %s", name[jj]), 7, -.5, 6.5);
      histLength1Fake[iH - 4][jj]->SetFillColor(colorArr[jj] - 9);
      histLength1Fake[iH - 4][jj]->SetLineColor(colorArr[jj] - 9);
      histLength1FakeNoCl[iH - 4][jj] = new TH1I(Form("trk_len_%d_1f_nocl_%s", iH, name[jj]), Form("#slash{#exists} cluster %s", name[jj]), 7, -.5, 6.5);
      histLength1FakeNoCl[iH - 4][jj]->SetFillColor(colorArr[jj] + 1);
      histLength1FakeNoCl[iH - 4][jj]->SetLineColor(colorArr[jj] + 1);
      if (jj == 0) {
        stackLength1Fake[iH - 4] = new THStack(Form("stack_trk_len_%d_1f", iH), Form("trk_len=%d, 1 Fake", iH));
      }
      stackLength1Fake[iH - 4]->Add(histLength1Fake[iH - 4][jj]);
      stackLength1Fake[iH - 4]->Add(histLength1FakeNoCl[iH - 4][jj]);

      histLength2Fake[iH - 4][jj] = new TH1I(Form("trk_len_%d_2f_%s", iH, name[jj]), Form("#exists cluster %s", name[jj]), 7, -.5, 6.5);
      histLength2Fake[iH - 4][jj]->SetFillColor(colorArr[jj] - 9);
      histLength2Fake[iH - 4][jj]->SetLineColor(colorArr[jj] - 9);
      histLength2FakeNoCl[iH - 4][jj] = new TH1I(Form("trk_len_%d_2f_nocl_%s", iH, name[jj]), Form("#slash{#exists} cluster %s", name[jj]), 7, -.5, 6.5);
      histLength2FakeNoCl[iH - 4][jj]->SetFillColor(colorArr[jj] + 1);
      histLength2FakeNoCl[iH - 4][jj]->SetLineColor(colorArr[jj] + 1);
      if (jj == 0) {
        stackLength2Fake[iH - 4] = new THStack(Form("stack_trk_len_%d_2f", iH), Form("trk_len=%d, 2 Fake", iH));
      }
      stackLength2Fake[iH - 4]->Add(histLength2Fake[iH - 4][jj]);
      stackLength2Fake[iH - 4]->Add(histLength2FakeNoCl[iH - 4][jj]);

      histLength3Fake[iH - 4][jj] = new TH1I(Form("trk_len_%d_3f_%s", iH, name[jj]), Form("#exists cluster %s", name[jj]), 7, -.5, 6.5);
      histLength3Fake[iH - 4][jj]->SetFillColor(colorArr[jj] - 9);
      histLength3Fake[iH - 4][jj]->SetLineColor(colorArr[jj] - 9);

      histLength3FakeNoCl[iH - 4][jj] = new TH1I(Form("trk_len_%d_3f_nocl_%s", iH, name[jj]), Form("#slash{#exists} cluster %s", name[jj]), 7, -.5, 6.5);
      histLength3FakeNoCl[iH - 4][jj]->SetFillColor(colorArr[jj] + 1);
      histLength3FakeNoCl[iH - 4][jj]->SetLineColor(colorArr[jj] + 1);
      if (jj == 0) {
        stackLength3Fake[iH - 4] = new THStack(Form("stack_trk_len_%d_3f", iH), Form("trk_len=%d, 3 Fake", iH));
      }
      stackLength3Fake[iH - 4]->Add(histLength3Fake[iH - 4][jj]);
      stackLength3Fake[iH - 4]->Add(histLength3FakeNoCl[iH - 4][jj]);
    }
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
  mInputITSidxs = recoData.getITSTracksClusterRefs();

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
        mParticleInfo[iSource][iEvent][iPart].prodProcessName = part.getProdProcessAsString();
        mParticleInfo[iSource][iEvent][iPart].prodProcess = part.getProcess();
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
  int unaccounted{0}, good{0}, fakes{0};
  // ***secondary tracks***
  int nPartForSpec[4][4];       // total number [particle 0=IperT, 1=Lambda, 2=k, 3=Other][n layer]
  int nPartGoodorFake[4][4][2]; // number of good or fake [particle 0=IperT, 1=Lambda, 2=k, 3=Other][n layer][good=1 fake=0]
  for (int n = 0; n < 4; n++) {
    for (int m = 0; m < 4; m++) {
      nPartForSpec[n][m] = 0;
      for (int h = 0; h < 2; h++) {
        nPartGoodorFake[n][m][h] = 0;
      }
    }
  }
  int nlayer = 999;
  int ngoodfake = 0;
  int totsec = 0;
  int totsecCont = 0;

  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
    auto& lab = mTracksMCLabels[iTrack];
    if (!lab.isSet() || lab.isNoise()) {
      unaccounted++;
      continue;
    }
    int trackID, evID, srcID;
    bool fake;
    const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);

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
  int totP{0}, goodP{0}, fakeP{0};
  // Currently process only sourceID = 0, to be extended later if needed
  for (auto& evInfo : mParticleInfo[0]) {
    trackID = 0;
    for (auto& part : evInfo) {

      if (strcmp(ProcessName[part.prodProcess], " ")) {
        ProcessName[part.prodProcess] = part.prodProcessName;
      }
      if ((part.clusters & 0x7f) == mMask) {
        // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
        // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters != 0x1f << 2 &&
        // part.clusters != 0x0f && part.clusters != 0x0f << 1 && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
        // continue;

        if (part.isPrimary) { // **Primary particle**
          totP++;
          mDenominatorPt->Fill(part.pt);
          mDenominatorEta->Fill(part.eta);
          if (part.isReco) {
            mGoodPt->Fill(part.pt);
            mGoodEta->Fill(part.eta);
            goodP++;
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
            fakeP++;
            if (part.isFake > 1) {
              for (int _i{0}; _i < part.isFake - 1; ++_i) {
                mMultiFake->Fill(part.pt);
              }
            }
          }
        }
      }

      // **Secondary particle**
      nlayer = 999;
      ngoodfake = 2;
      if (!part.isPrimary) {
        int TrackID, EvID, SrcID;
        int pdgcode = mParticleInfo[0][evID][part.mother].pdg;
        int idxPart = 999;
        float rad = sqrt(pow(part.vx, 2) + pow(part.vy, 2));
        totsec++;

        if ((rad < rLayer0) && (part.clusters == 0x7f || part.clusters == 0x3f || part.clusters == 0x1f || part.clusters == 0x0f)) { // layer 0
          nlayer = 0;
        }
        if (rad < rLayer1 && rad > rLayer0 && (part.clusters == 0x1e || part.clusters == 0x3e || part.clusters == 0x7e)) { // layer 1
          nlayer = 1;
        }
        if (rad < rLayer2 && rad > rLayer1 && (part.clusters == 0x7c || part.clusters == 0x3c)) { // layer 2
          nlayer = 2;
        }
        if (rad < rLayer3 && rad > rLayer2 && part.clusters == 0x78) { // layer 3
          nlayer = 3;
        }
        if (nlayer == 0 || nlayer == 1 || nlayer == 2 || nlayer == 3) { // check if track is trackeable

          totsecCont++;
          processvsZ->Fill(part.vz, part.prodProcess);
          processvsRad->Fill(rad, part.prodProcess);
          mDenominatorPtSec->Fill(part.pt);
          mDenominatorEtaSec->Fill(part.eta);
          mTotRad[3]->Fill(rad);
          mTotZ[3]->Fill(part.vz);
          mTotPts[nlayer][3]->Fill(part.pt);
          mTotEtas[nlayer][3]->Fill(part.eta);
          mTotPts[nlayer][3]->Fill(part.pt);
          mTotEtas[nlayer][3]->Fill(part.eta);
          if (pdgcode == PDG[0] || pdgcode == -1 * PDG[0]) {
            idxPart = 0; // IperT
          }
          if (pdgcode == PDG[1] || pdgcode == -1 * PDG[1]) {
            idxPart = 1; // Lambda
          }
          if (pdgcode == PDG[2] || pdgcode == -1 * PDG[2]) {
            idxPart = 2; // K0s
          }
          if (part.isReco) {
            ngoodfake = 1;
            mGoodPts[3][nlayer]->Fill(part.pt);
            mGoodEtas[3][nlayer]->Fill(part.eta);
            mGoodPtSec->Fill(part.pt);
            mGoodEtaSec->Fill(part.eta);
            mGoodRad[3]->Fill(rad);
            mGoodZ[3]->Fill(part.vz);
          }
          if (part.isFake) {
            ngoodfake = 0;
            mFakePts[3][nlayer]->Fill(part.pt);
            mFakeEtas[3][nlayer]->Fill(part.eta);
            mFakePtSec->Fill(part.pt);
            mFakeEtaSec->Fill(part.eta);
            mFakeRad[3]->Fill(rad);
            mFakeZ[3]->Fill(part.vz);
          }
          if (idxPart < 3) // to change if the number of analysing particle changes
          {
            mTotRad[idxPart]->Fill(rad);
            mTotZ[idxPart]->Fill(part.vz);
            mTotPts[idxPart][nlayer]->Fill(part.pt);
            mTotEtas[idxPart][nlayer]->Fill(part.eta);
            if (part.isReco) {
              mGoodRad[idxPart]->Fill(rad);
              mGoodZ[idxPart]->Fill(part.vz);
              mGoodPts[idxPart][nlayer]->Fill(part.pt);
              mGoodEtas[idxPart][nlayer]->Fill(part.eta);
            }
            if (part.isFake) {
              mFakeRad[idxPart]->Fill(rad);
              mFakeZ[idxPart]->Fill(part.vz);
              mFakePts[idxPart][nlayer]->Fill(part.pt);
              mFakeEtas[idxPart][nlayer]->Fill(part.eta);
            }
          }

          if (pdgcode != 1010010030 && pdgcode != 3122 && pdgcode != 310 && pdgcode != -1010010030 && pdgcode != -310 && pdgcode != -3122) {
            idxPart = 3;
            processvsRadOther->Fill(rad, part.prodProcess);
          }

          if (!part.isFake && !part.isReco) {
            processvsEtaNotTracked->Fill(part.eta, part.prodProcess);
            processvsRadNotTracked->Fill(rad, part.prodProcess);
          }
          if (ngoodfake == 1 || ngoodfake == 0) {
            nPartGoodorFake[idxPart][nlayer][ngoodfake]++;
          }
          nPartForSpec[idxPart][nlayer]++;

          // Analysing fake clusters
          int nCl{0};
          for (unsigned int bit{0}; bit < sizeof(part.clusters) * 8; ++bit) {
            nCl += bool(part.clusters & (1 << bit));
          }
          if (nCl < 3) {
            continue;
          }
          if (idxPart < 3) {
            auto& track = part.track;
            auto len = track.getNClusters();
            int nclu = track.getNumberOfClusters();
            int firstclu = track.getFirstClusterEntry();
            for (int iLayer{0}; iLayer < 7; ++iLayer) {
              if (track.hasHitOnLayer(iLayer)) {
                if (track.isFakeOnLayer(iLayer)) {
                  // Reco track has fake cluster
                  if (part.clusters & (0x1 << iLayer)) { // Correct cluster exists
                    histLength[len - 4][idxPart]->Fill(iLayer);
                    if (track.getNFakeClusters() == 1) {
                      histLength1Fake[len - 4][idxPart]->Fill(iLayer);
                    }
                    if (track.getNFakeClusters() == 2) {
                      histLength2Fake[len - 4][idxPart]->Fill(iLayer);
                    }
                    if (track.getNFakeClusters() == 3) {
                      histLength3Fake[len - 4][idxPart]->Fill(iLayer);
                    }
                  } else {

                    histLengthNoCl[len - 4][idxPart]->Fill(iLayer);
                    if (track.getNFakeClusters() == 1) {
                      histLength1FakeNoCl[len - 4][idxPart]->Fill(iLayer);
                    }
                    if (track.getNFakeClusters() == 2) {
                      histLength2FakeNoCl[len - 4][idxPart]->Fill(iLayer);
                    }
                    if (track.getNFakeClusters() == 3) {
                      histLength3FakeNoCl[len - 4][idxPart]->Fill(iLayer);
                    }
                  }
                  auto labs = mClustersMCLCont->getLabels(mInputITSidxs[firstclu - 1 - iLayer + track.getFirstClusterLayer() + nclu]);

                  for (auto& lab : labs) {
                    if (!lab.isValid()) {
                      continue; // We want to skip channels related to noise, e.g. sID = 99: QED
                    }

                    bool fakec;
                    const_cast<o2::MCCompLabel&>(lab).get(TrackID, EvID, SrcID, fakec);
                    double intHisto = 0;
                    for (int hg = 0; hg < 7; hg++) {
                      if (mParticleInfo[SrcID][EvID][TrackID].pdg == PdgcodeClusterFake[hg] || mParticleInfo[SrcID][EvID][TrackID].pdg == -1 * (PdgcodeClusterFake[hg])) {
                        intHisto = hg + 0.5;
                      }
                    }
                    if (idxPart < 3) {
                      mClusterFake[idxPart]->Fill(intHisto, mParticleInfo[SrcID][EvID][TrackID].prodProcess);
                    }
                  }
                }
              }
            }
          }
        }
        nlayer = 999;
      }
      trackID++;
    }
    evID++;
  }

  int totgood{0}, totfake{0}, totI{0}, totL{0}, totK{0}, totO{0};
  for (int xx = 0; xx < 4; xx++) {
    for (int yy = 0; yy < 4; yy++) {
      totgood = totgood + nPartGoodorFake[xx][yy][1];
      totfake = totfake + nPartGoodorFake[xx][yy][0];
      if (xx == 0) {
        totI = totI + nPartForSpec[0][yy];
      }
      if (xx == 1) {
        totL = totL + nPartForSpec[1][yy];
      }
      if (xx == 2) {
        totK = totK + nPartForSpec[2][yy];
      }
      if (xx == 3) {
        totO = totO + nPartForSpec[3][yy];
      }
    }
  }
  LOGP(info, "number of primary tracks: {}, good:{}, fake:{}", totP, goodP, fakeP);
  int goodI = nPartGoodorFake[0][0][1] + nPartGoodorFake[0][1][1] + nPartGoodorFake[0][2][1] + nPartGoodorFake[0][3][1];
  int goodL = nPartGoodorFake[1][0][1] + nPartGoodorFake[1][1][1] + nPartGoodorFake[1][2][1] + nPartGoodorFake[1][3][1];
  int goodK = nPartGoodorFake[2][0][1] + nPartGoodorFake[2][1][1] + nPartGoodorFake[2][2][1] + nPartGoodorFake[2][3][1];
  int fakeI = nPartGoodorFake[0][0][0] + nPartGoodorFake[0][1][0] + nPartGoodorFake[0][2][0] + nPartGoodorFake[0][3][0];
  int fakeL = nPartGoodorFake[1][0][0] + nPartGoodorFake[1][1][0] + nPartGoodorFake[1][2][0] + nPartGoodorFake[1][3][0];
  int fakeK = nPartGoodorFake[2][0][0] + nPartGoodorFake[2][1][0] + nPartGoodorFake[2][2][0] + nPartGoodorFake[2][3][0];
  LOGP(info, "** Some statistics on secondary tracks:");

  LOGP(info, "\t- Total number of secondary tracks: {}", totsec);
  LOGP(info, "\t- Total number of secondary trackeable tracks : {}", totsecCont);
  LOGP(info, "\t- Total number of secondary trackeable tracks good: {}, fake: {}", totgood, totfake);
  LOGP(info, "\t- Total number of secondary trackeable tracks from IperT: {} = {} %, Good={} % , fake={} %", totI, 100 * totI / totsecCont, 100 * goodI / totI, 100 * fakeI / totI);
  LOGP(info, "\t- Total number of secondary trackeable tracks from Lam: {} = {} %, Good={} % , fake={} %", totL, 100 * totL / totsecCont, 100 * goodL / totL, 100 * fakeL / totL);
  LOGP(info, "\t- Total number of secondary trackeable tracks from k: {} = {} %, Good={} % , fake={} %", totK, 100 * totK / totsecCont, 100 * goodK / totK, 100 * fakeK / totK);
  LOGP(info, "\t- Total number of secondary trackeable tracks from Other: {} = {} %", totO, 100 * totO / totsecCont);

  LOGP(info, "** Computing efficiencies ...");

  mEffPt = std::make_unique<TEfficiency>(*mGoodPt, *mDenominatorPt);
  mEffFakePt = std::make_unique<TEfficiency>(*mFakePt, *mDenominatorPt);
  mEffClonesPt = std::make_unique<TEfficiency>(*mClonePt, *mDenominatorPt);

  mEffEta = std::make_unique<TEfficiency>(*mGoodEta, *mDenominatorEta);
  mEffFakeEta = std::make_unique<TEfficiency>(*mFakeEta, *mDenominatorEta);
  mEffClonesEta = std::make_unique<TEfficiency>(*mCloneEta, *mDenominatorEta);

  mEffPtSec = std::make_unique<TEfficiency>(*mGoodPtSec, *mDenominatorPtSec);
  mEffFakePtSec = std::make_unique<TEfficiency>(*mFakePtSec, *mDenominatorPtSec);

  mEffEtaSec = std::make_unique<TEfficiency>(*mGoodEtaSec, *mDenominatorEtaSec);
  mEffFakeEtaSec = std::make_unique<TEfficiency>(*mFakeEtaSec, *mDenominatorEtaSec);

  for (int ii = 0; ii < 4; ii++) {
    for (int yy = 0; yy < 4; yy++) {
      mEffGoodPts[ii][yy] = std::make_unique<TEfficiency>(*mGoodPts[ii][yy], *mTotPts[ii][yy]);
      mEffFakePts[ii][yy] = std::make_unique<TEfficiency>(*mFakePts[ii][yy], *mTotPts[ii][yy]);
      mEffGoodEtas[ii][yy] = std::make_unique<TEfficiency>(*mGoodEtas[ii][yy], *mTotEtas[ii][yy]);
      mEffFakeEtas[ii][yy] = std::make_unique<TEfficiency>(*mFakeEtas[ii][yy], *mTotEtas[ii][yy]);
    }
    mEffGoodRad[ii] = std::make_unique<TEfficiency>(*mGoodRad[ii], *mTotRad[ii]);
    mEffFakeRad[ii] = std::make_unique<TEfficiency>(*mFakeRad[ii], *mTotRad[ii]);
    mEffGoodZ[ii] = std::make_unique<TEfficiency>(*mGoodZ[ii], *mTotZ[ii]);
    mEffFakeZ[ii] = std::make_unique<TEfficiency>(*mFakeZ[ii], *mTotZ[ii]);
  }

  LOGP(info, "** Analysing pT resolution...");
  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
    auto& lab = mTracksMCLabels[iTrack];
    if (!lab.isSet() || lab.isNoise()) {
      continue;
    }
    int trackID, evID, srcID;
    bool fake;
    const_cast<o2::MCCompLabel&>(lab).get(trackID, evID, srcID, fake);
    if (srcID == 99) {
      continue; // skip QED
    }
    mPtResolution->Fill((mParticleInfo[srcID][evID][trackID].pt - mTracks[iTrack].getPt()) / mParticleInfo[srcID][evID][trackID].pt);
    mPtResolution2D->Fill(mParticleInfo[srcID][evID][trackID].pt, (mParticleInfo[srcID][evID][trackID].pt - mTracks[iTrack].getPt()) / mParticleInfo[srcID][evID][trackID].pt);
    if (!mParticleInfo[srcID][evID][trackID].isPrimary) {
      mPtResolutionSec->Fill((mParticleInfo[srcID][evID][trackID].pt - mTracks[iTrack].getPt()) / mParticleInfo[srcID][evID][trackID].pt);
    }
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

  setEfficiencyGraph(mEffPt, "Good_pt", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kAzure + 4, 0.65);
  fout.WriteTObject(mEffPt.get());

  setEfficiencyGraph(mEffFakePt, "Fake_pt", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kRed, 0.65);
  fout.WriteTObject(mEffFakePt.get());

  setEfficiencyGraph(mEffPtSec, "Good_ptSec", ";#it{p}_{T} (GeV/#it{c});efficiency secondary particle", kOrange + 7);
  fout.WriteTObject(mEffPtSec.get());

  setEfficiencyGraph(mEffFakePtSec, "Fake_ptSec", ";#it{p}_{T} (GeV/#it{c});efficiency secondary particle", kGray + 2);
  fout.WriteTObject(mEffFakePtSec.get());

  setEfficiencyGraph(mEffClonesPt, "Clone_pt", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kGreen + 2, 0.65);
  fout.WriteTObject(mEffClonesPt.get());

  setEfficiencyGraph(mEffEta, "Good_eta", ";#eta;efficiency primary particle", kAzure + 4, 0.65);
  fout.WriteTObject(mEffEta.get());

  setEfficiencyGraph(mEffFakeEta, "Fake_eta", ";#eta;efficiency primary particle", kRed + 1, 0.65);
  fout.WriteTObject(mEffFakeEta.get());

  setEfficiencyGraph(mEffEtaSec, "Good_etaSec", ";#eta;efficiency secondary particle", kOrange + 7);
  fout.WriteTObject(mEffEtaSec.get());

  setEfficiencyGraph(mEffFakeEtaSec, "Fake_etaSec", ";#eta;efficiency secondary particle", kGray + 2);
  fout.WriteTObject(mEffFakeEtaSec.get());

  setEfficiencyGraph(mEffClonesEta, "Clone_eta", ";#it{p}_{T} (GeV/#it{c});efficiency primary particle", kGreen + 2, 0.65);
  fout.WriteTObject(mEffClonesEta.get());

  for (int aa = 0; aa < 4; aa++) {
    setEfficiencyGraph(mEffGoodRad[aa], Form("Good_Rad_%s", particleToanalize[aa]), ";Radius [cm];efficiency secondary particle", colorArr[aa]);
    fout.WriteTObject(mEffGoodRad[aa].get());

    setEfficiencyGraph(mEffGoodRad[aa], Form("Fake_Rad_%s", particleToanalize[aa]), ";Radius [cm];efficiency secondary particle", colorArr[aa] - 9);
    fout.WriteTObject(mEffGoodRad[aa].get());

    setEfficiencyGraph(mEffGoodZ[aa], Form("Good_Z_%s", particleToanalize[aa]), ";Z_{sv} [cm];efficiency secondary particle", colorArr[aa]);
    fout.WriteTObject(mEffGoodZ[aa].get());

    setEfficiencyGraph(mEffGoodZ[aa], Form("Fake_Z_%s", particleToanalize[aa]), ";Z_{sv} [cm];efficiency secondary particle", colorArr[aa] - 9);
    fout.WriteTObject(mEffGoodZ[aa].get());

    for (int bb = 0; bb < 4; bb++) {
      setEfficiencyGraph(mEffGoodPts[aa][bb], Form("EffPtGood_%sl%d", particleToanalize[aa], bb), Form("Good Sec Tracks_%s, L%d"
                                                                                                       ";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ",
                                                                                                       particleToanalize[aa], bb),
                         colorArr[aa]);
      setEfficiencyGraph(mEffFakePts[aa][bb], Form("EffPtFake_%sl%d", particleToanalize[aa], bb), Form("Fake Sec Tracks_%s, L%d"
                                                                                                       ";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ",
                                                                                                       particleToanalize[aa], bb),
                         colorArr[aa]);
      setEfficiencyGraph(mEffGoodEtas[aa][bb], Form("EffEtaGood_%sl%d", particleToanalize[aa], bb), Form("Good Sec Tracks_%s, L%d"
                                                                                                         ";#eta ;efficiency secondary particle ",
                                                                                                         particleToanalize[aa], bb),
                         colorArr[aa]);
      setEfficiencyGraph(mEffFakeEtas[aa][bb], Form("EffEtaFake_%sl%d", particleToanalize[aa], bb), Form("Fake Sec Tracks_%s, L%d"
                                                                                                         ";#eta ;efficiency secondary particle ",
                                                                                                         particleToanalize[aa], bb),
                         colorArr[aa]);

      fout.WriteTObject(mEffGoodPts[aa][bb].get());
      fout.WriteTObject(mEffFakePts[aa][bb].get());
      fout.WriteTObject(mEffGoodEtas[aa][bb].get());
      fout.WriteTObject(mEffFakeEtas[aa][bb].get());
    }
    for (int i = 0; i < 3; i++) {
      fout.WriteTObject(histLength[aa][i], Form("trk_len_%d_%s", 4 + aa, name[i]));
      fout.WriteTObject(histLength1Fake[aa][i], Form("trk_len_%d_1f_%s", 4 + aa, name[i]));
      fout.WriteTObject(histLength2Fake[aa][i], Form("trk_len_%d_2f_%s", 4 + aa, name[i]));
      fout.WriteTObject(histLength3Fake[aa][i], Form("trk_len_%d_3f_%s", 4 + aa, name[i]));
      fout.WriteTObject(histLengthNoCl[aa][i], Form("trk_len_%d_nocl_%s", 4 + aa, name[i]));
      fout.WriteTObject(histLength1FakeNoCl[aa][i], Form("trk_len_%d_1f_nocl_%s", 4 + aa, name[i]));
      fout.WriteTObject(histLength2FakeNoCl[aa][i], Form("trk_len_%d_2f_nocl_%s", 4 + aa, name[i]));
      fout.WriteTObject(histLength3FakeNoCl[aa][i], Form("trk_len_%d_3f_nocl_%s", 4 + aa, name[i]));
    }
  }

  for (int j = 0; j < 4; j++) {
    for (int i = 1; i <= 7; i++) {
      mClusterFake[j]->GetXaxis()->SetBinLabel(i, ParticleName[i - 1]);

      for (int i = 1; i <= 50; i++) {
        mClusterFake[j]->GetYaxis()->SetBinLabel(i, ProcessName[i - 1]);
        if (j == 0) {
          processvsZ->GetYaxis()->SetBinLabel(i, ProcessName[i - 1]);
          processvsRad->GetYaxis()->SetBinLabel(i, ProcessName[i - 1]);
          processvsRadOther->GetYaxis()->SetBinLabel(i, ProcessName[i - 1]);
          processvsRadNotTracked->GetYaxis()->SetBinLabel(i, ProcessName[i - 1]);
          processvsEtaNotTracked->GetYaxis()->SetBinLabel(i, ProcessName[i - 1]);
        }
      }
      fout.WriteTObject(mClusterFake[j].get());
    }
  }
  fout.WriteTObject(processvsZ.get());
  fout.WriteTObject(processvsRad.get());
  fout.WriteTObject(processvsRadOther.get());
  fout.WriteTObject(processvsRadNotTracked.get());
  fout.WriteTObject(processvsEtaNotTracked.get());

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
  mLegendPt->SetHeader(Form("%zu events PP min bias", mKineReader->getNEvents(0)), "C");
  mLegendPt->AddEntry("Good_pt", "good (100% cluster purity)", "lep");
  mLegendPt->AddEntry("Fake_pt", "fake", "lep");
  mLegendPt->AddEntry("Clone_pt", "clone", "lep");
  mLegendPt->Draw();
  mCanvasPt->SaveAs("eff_pt.png");

  mCanvasPtSec = std::make_unique<TCanvas>("cPtSec", "cPtSec", 1600, 1200);
  mCanvasPtSec->cd();
  mCanvasPtSec->SetLogx();
  mCanvasPtSec->SetGrid();
  mEffPtSec->Draw("pz");
  mEffFakePtSec->Draw("pz same");
  mLegendPtSec = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendPtSec->SetHeader(Form("%zu events PP min bias", mKineReader->getNEvents(0)), "C");
  mLegendPtSec->AddEntry("Good_ptSec", "good (100% cluster purity)", "lep");
  mLegendPtSec->AddEntry("Fake_tSec", "fake", "lep");
  mLegendPtSec->Draw();
  mCanvasPtSec->SaveAs("eff_ptSec.png");

  mCanvasEta = std::make_unique<TCanvas>("cEta", "cEta", 1600, 1200);
  mCanvasEta->cd();
  mCanvasEta->SetGrid();
  mEffEta->Draw("pz");
  mEffFakeEta->Draw("pz same");
  mEffClonesEta->Draw("pz same");
  mLegendEta = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendEta->SetHeader(Form("%zu events PP min bias", mKineReader->getNEvents(0)), "C");
  mLegendEta->AddEntry("Good_eta", "good (100% cluster purity)", "lep");
  mLegendEta->AddEntry("Fake_eta", "fake", "lep");
  mLegendEta->AddEntry("Clone_eta", "clone", "lep");
  mLegendEta->Draw();
  mCanvasEta->SaveAs("eff_eta.png");

  mCanvasEtaSec = std::make_unique<TCanvas>("cEtaSec", "cEtaSec", 1600, 1200);
  mCanvasEtaSec->cd();
  mCanvasEtaSec->SetGrid();
  mEffEtaSec->Draw("pz");
  mEffFakeEtaSec->Draw("pz same");
  mLegendEtaSec = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendEtaSec->SetHeader(Form("%zu events PP min bias", mKineReader->getNEvents(0)), "C");
  mLegendEtaSec->AddEntry("Good_etaSec", "good (100% cluster purity)", "lep");
  mLegendEtaSec->AddEntry("Fake_etaSec", "fake", "lep");
  mLegendEtaSec->Draw();
  mCanvasEtaSec->SaveAs("eff_EtaSec.png");

  mCanvasRad = std::make_unique<TCanvas>("cRad", "cRad", 1600, 1200);
  mCanvasRad->cd();
  mCanvasRad->SetGrid();
  mEffGoodRad[3]->Draw("pz");
  mEffFakeRad[3]->Draw("pz same");
  mCanvasRad->SetLogy();
  mLegendRad = std::make_unique<TLegend>(0.8, 0.4, 0.95, 0.6);
  mLegendRad->SetHeader(Form("%zu events PP ", mKineReader->getNEvents(0)), "C");
  mLegendRad->AddEntry(Form("Good_Rad_%s", particleToanalize[3]), "good", "lep");
  mLegendRad->AddEntry(Form("Fake_Rad_%s", particleToanalize[3]), "fake", "lep");
  mLegendRad->Draw();
  mCanvasRad->SaveAs("eff_rad_sec.png");

  mCanvasZ = std::make_unique<TCanvas>("cZ", "cZ", 1600, 1200);
  mCanvasZ->cd();
  mCanvasZ->SetGrid();
  mCanvasZ->SetLogy();
  mEffGoodZ[3]->Draw("pz");
  mEffFakeZ[3]->Draw("pz same");
  mCanvasZ->SetLogy();
  mLegendZ = std::make_unique<TLegend>(0.8, 0.4, 0.95, 0.6);
  mLegendZ->SetHeader(Form("%zu events PP ", mKineReader->getNEvents(0)), "C");
  mLegendZ->AddEntry(Form("Good_Z_%s", particleToanalize[3]), "good", "lep");
  mLegendZ->AddEntry(Form("Fake_Z_%s", particleToanalize[3]), "fake", "lep");
  mLegendZ->Draw();
  mCanvasZ->SaveAs("eff_Z_sec.png");
  ;

  mCanvasRadD = std::make_unique<TCanvas>("cRadD", "cRadD", 1600, 1200);
  mCanvasRadD->cd();
  mCanvasRadD->SetGrid();
  mCanvasRadD->SetLogy();
  mLegendRadD = std::make_unique<TLegend>(0.8, 0.64, 0.95, 0.8);
  mLegendRadD->SetHeader(Form("%zu events PP ", mKineReader->getNEvents(0)), "C");
  for (int i = 0; i < 3; i++) {
    if (i == 0) {
      mEffGoodRad[i]->Draw("pz");
    } else {
      mEffGoodRad[i]->Draw("pz same");
      mEffFakeRad[i]->Draw("pz same");
      mLegendRadD->AddEntry(Form("Good_Rad%s", particleToanalize[i]), Form("%s_good", name[i]), "lep");
      mLegendRadD->AddEntry(Form("Fake_Rad%s", particleToanalize[i]), Form("%s_fake", name[i]), "lep");
    }
  }
  mLegendRadD->Draw();
  mCanvasRadD->SaveAs("eff_RadD_sec.png");

  mCanvasZD = std::make_unique<TCanvas>("cZD", "cZD", 1600, 1200);
  mCanvasZD->cd();
  mCanvasZD->SetGrid();
  mCanvasZD->SetLogy();
  mLegendZD = std::make_unique<TLegend>(0.8, 0.64, 0.95, 0.8);
  mLegendZD->SetHeader(Form("%zu events PP ", mKineReader->getNEvents(0)), "C");
  for (int i = 0; i < 3; i++) {
    if (i == 0) {
      mEffGoodZ[i]->Draw("pz");
    } else {
      mEffGoodZ[i]->Draw("pz same");
      mEffFakeZ[i]->Draw("pz same");
      mLegendZD->AddEntry(Form("Good_Z%s", particleToanalize[i]), Form("%s_good", name[i]), "lep");
      mLegendZD->AddEntry(Form("Fake_Z%s", particleToanalize[i]), Form("%s_fake", name[i]), "lep");
    }
  }
  mLegendZD->Draw();
  mCanvasZD->SaveAs("eff_ZD_sec.png");

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

  auto* g1 = new TGraphErrors(100, meanPt, sigma, aa, sigmaerr);
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
    stackLength[iH]->GetXaxis()->SetTitle("Layer");
    gPad->BuildLegend();
  }
  for (int iH{0}; iH < 4; ++iH) {
    canvas->cd(iH + 5);
    stackLength1Fake[iH]->Draw();
    stackLength1Fake[iH]->GetXaxis()->SetTitle("Layer");
    gPad->BuildLegend();
  }

  canvas->SaveAs("fakeClusters2.png", "recreate");

  auto canvas2 = new TCanvas("fc_canvas2", "Fake clusters", 1600, 1000);
  canvas2->Divide(4, 2);

  for (int iH{0}; iH < 4; ++iH) {
    canvas2->cd(iH + 1);
    stackLength2Fake[iH]->Draw();
    stackLength2Fake[iH]->GetXaxis()->SetTitle("Layer");
    gPad->BuildLegend();
  }
  for (int iH{0}; iH < 4; ++iH) {
    canvas2->cd(iH + 5);
    stackLength3Fake[iH]->Draw();
    stackLength3Fake[iH]->GetXaxis()->SetTitle("Layer");
    gPad->BuildLegend();
  }
  canvas2->SaveAs("fakeClusters3.png", "recreate");

  auto canvasPtfake = new TCanvas("canvasPtfake", "Fake pt", 1600, 1000);
  canvasPtfake->Divide(2, 2);

  for (int iH{0}; iH < 4; ++iH) {
    canvasPtfake->cd(iH + 1);
    for (int v = 0; v < 4; v++) {
      if (v == 0) {
        canvasPtfake->cd(iH + 1);
      }
      if (v == 0) {
        mEffFakePts[v][iH]->Draw();
      } else {
        mEffFakePts[v][iH]->Draw("same");
      }
    }
    gPad->BuildLegend();
    gPad->SetGrid();
    gPad->SetTitle(Form("#it{p}_{T}, Fake Tracks, layer %d", iH));
    gPad->SetName(Form("#it{p}_{T}, Fake Tracks, layer %d", iH));
  }
  canvasPtfake->SaveAs("PtforPartFake.png", "recreate");

  auto canvasPtGood = new TCanvas("canvasPtGood", "Good pt", 1600, 1000);
  canvasPtGood->Divide(2, 2);

  for (int iH{0}; iH < 4; ++iH) {
    canvasPtGood->cd(iH + 1);
    for (int v = 0; v < 4; v++) {
      if (v == 0) {
        canvasPtGood->cd(iH + 1);
      }
      if (v == 0) {
        mEffGoodPts[v][iH]->Draw();
      } else {
        mEffGoodPts[v][iH]->Draw("same");
      }
    }
    gPad->BuildLegend();
    gPad->SetGrid();
    gPad->SetTitle(Form("#it{p}_{T}, Good Tracks, layer %d", iH));
    gPad->SetName(Form("#it{p}_{T}, Good Tracks, layer %d", iH));
  }

  auto canvasEtafake = new TCanvas("canvasEtafake", "Fake Eta", 1600, 1000);
  canvasEtafake->Divide(2, 2);

  for (int iH{0}; iH < 4; ++iH) {
    canvasEtafake->cd(iH + 1);
    for (int v = 0; v < 4; v++) {
      if (v == 0) {
        canvasEtafake->cd(iH + 1);
      }
      if (v == 0) {
        mEffFakeEtas[v][iH]->Draw();
      } else {
        mEffFakeEtas[v][iH]->Draw("same");
      }
    }
    gPad->BuildLegend();
    gPad->SetGrid();
    gPad->SetTitle(Form("#eta, Fake Tracks, layer %d", iH));
    gPad->SetName(Form("#eta, Fake Tracks, layer %d", iH));
  }
  auto canvasEtaGood = new TCanvas("canvasEtaGood", "Good Eta", 1600, 1000);
  canvasEtaGood->Divide(2, 2);

  for (int iH{0}; iH < 4; ++iH) {
    canvasEtaGood->cd(iH + 1);
    for (int v = 0; v < 4; v++) {
      if (v == 0) {
        canvasEtaGood->cd(iH + 1);
      }
      if (v == 0) {
        mEffGoodEtas[v][iH]->Draw();
      } else {
        mEffGoodEtas[v][iH]->Draw("same");
      }
    }
    gPad->BuildLegend();
    gPad->SetGrid();
    gPad->SetTitle(Form("#eta, Good Tracks, layer %d", iH));
    gPad->SetName(Form("#eta, Good Tracks, layer %d", iH));
  }

  auto canvasI = new TCanvas("canvasI", "canvasI", 1600, 1000);
  canvasI->cd();
  mClusterFake[0]->Draw("COLZ");
  canvasI->SaveAs("Iper2D.png", "recreate");

  auto canvasL = new TCanvas("canvasL", "canvasL", 1600, 1000);
  canvasL->cd();
  mClusterFake[1]->Draw("COLZ");
  canvasL->SaveAs("Lam2D.png", "recreate");

  auto canvasK = new TCanvas("canvasK", "canvasK", 1600, 1000);
  canvasK->cd();
  mClusterFake[2]->Draw("COLZ");
  canvasK->SaveAs("K2D.png", "recreate");

  auto canvasZProd = new TCanvas("canvasZProd", "canvasZProd", 1600, 1000);
  canvasZProd->cd();
  processvsZ->Draw("COLZ");
  canvasZProd->SaveAs("prodvsZ.png", "recreate");
  auto canvasRadProd = new TCanvas("canvasRadProd", "canvasRadProd", 1600, 1000);
  canvasRadProd->cd();
  processvsRad->Draw("COLZ");
  canvasRadProd->SaveAs("prodvsRad.png", "recreate");
  auto canvasRadProdO = new TCanvas("canvasRadProdO", "canvasRadProdO", 1600, 1000);
  canvasRadProdO->cd();
  processvsRadOther->Draw("COLZ");
  canvasRadProdO->SaveAs("prodvsRadO.png", "recreate");
  auto canvasRadProNOTr = new TCanvas("canvasRadProNOTr", "canvasRadProNOTr", 1600, 1000);
  canvasRadProNOTr->cd();
  processvsRadNotTracked->Draw("COLZ");
  canvasRadProNOTr->SaveAs("prodvsRadNoTr.png", "recreate");
  auto canvasEtaProNOTr = new TCanvas("canvasEtaProNOTr", "canvasEtaProNOTr", 1600, 1000);
  canvasEtaProNOTr->cd();
  processvsEtaNotTracked->Draw("COLZ");
  canvasEtaProNOTr->SaveAs("prodvsEtaNoTr.png", "recreate");

  fout.cd();
  mCanvasPt->Write();
  mCanvasEta->Write();
  mCanvasPtSec->Write();
  mCanvasEtaSec->Write();
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
  canvasI->Write();
  canvasL->Write();
  canvasK->Write();
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

} // namespace o2::its::study
