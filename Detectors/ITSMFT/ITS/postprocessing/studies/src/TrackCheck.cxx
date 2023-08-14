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
#include <TCanvas.h>
#include <TEfficiency.h>
#include <TStyle.h>
#include <TLegend.h>

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
    unsigned short clusters = 0u;
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

  std::unique_ptr<TEfficiency> mEffPt;
  std::unique_ptr<TEfficiency> mEffFakePt;
  std::unique_ptr<TEfficiency> mEffClonesPt;

  std::unique_ptr<TEfficiency> mEffEta;
  std::unique_ptr<TEfficiency> mEffFakeEta;
  std::unique_ptr<TEfficiency> mEffClonesEta;

  std::unique_ptr<TEfficiency> mEff0Pt;
  std::unique_ptr<TEfficiency> mEff1Pt;
  std::unique_ptr<TEfficiency> mEff2Pt;
  std::unique_ptr<TEfficiency> mEff3Pt;

  std::unique_ptr<TEfficiency> mEff0FakePt;
  std::unique_ptr<TEfficiency> mEff1FakePt;
  std::unique_ptr<TEfficiency> mEff2FakePt;
  std::unique_ptr<TEfficiency> mEff3FakePt;

  std::unique_ptr<TEfficiency> mEff0Eta;
  std::unique_ptr<TEfficiency> mEff1Eta;
  std::unique_ptr<TEfficiency> mEff2Eta;
  std::unique_ptr<TEfficiency> mEff3Eta;

  std::unique_ptr<TEfficiency> mEff0FakeEta;
  std::unique_ptr<TEfficiency> mEff1FakeEta;
  std::unique_ptr<TEfficiency> mEff2FakeEta;
  std::unique_ptr<TEfficiency> mEff3FakeEta;

  std::unique_ptr<TH1D> mGoodPt0;
  std::unique_ptr<TH1D> mGoodPt1;
  std::unique_ptr<TH1D> mGoodPt2;
  std::unique_ptr<TH1D> mGoodPt3;

  std::unique_ptr<TH1D> mFakePt0;
  std::unique_ptr<TH1D> mFakePt1;
  std::unique_ptr<TH1D> mFakePt2;
  std::unique_ptr<TH1D> mFakePt3;

  std::unique_ptr<TH1D> mGoodEta0;
  std::unique_ptr<TH1D> mGoodEta1;
  std::unique_ptr<TH1D> mGoodEta2;
  std::unique_ptr<TH1D> mGoodEta3;

  std::unique_ptr<TH1D> mFakeEta0;
  std::unique_ptr<TH1D> mFakeEta1;
  std::unique_ptr<TH1D> mFakeEta2;
  std::unique_ptr<TH1D> mFakeEta3;

  std::unique_ptr<TH1D> mPtSec0Pt;
  std::unique_ptr<TH1D> mPtSec1Pt;
  std::unique_ptr<TH1D> mPtSec2Pt;
  std::unique_ptr<TH1D> mPtSec3Pt;

  std::unique_ptr<TH1D> mPtSec0Eta;
  std::unique_ptr<TH1D> mPtSec1Eta;
  std::unique_ptr<TH1D> mPtSec2Eta;
  std::unique_ptr<TH1D> mPtSec3Eta;

  // Canvas & decorations
  std::unique_ptr<TCanvas> mCanvasPt;
  std::unique_ptr<TCanvas> mCanvasPt2;
  std::unique_ptr<TCanvas> mCanvasPt2fake;
  std::unique_ptr<TCanvas> mCanvasPtRes;
  std::unique_ptr<TCanvas> mCanvasEta;
  std::unique_ptr<TCanvas> mCanvasEta2;
  std::unique_ptr<TCanvas> mCanvasEta2fake;
  std::unique_ptr<TLegend> mLegendPt;
  std::unique_ptr<TLegend> mLegendPt2;
  std::unique_ptr<TLegend> mLegendPt2Fake;
  std::unique_ptr<TLegend> mLegendEta;
  std::unique_ptr<TLegend> mLegendEta2;
  std::unique_ptr<TLegend> mLegendEta2Fake;

  float rLayer0 = 2.34;
  float rLayer1 = 3.15;
  float rLayer2 = 3.93;
  float rLayer3 = 19.605;

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

  mGoodPt = std::make_unique<TH1D>("goodPt", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
  mGoodEta = std::make_unique<TH1D>("goodEta", ";#eta;Number of tracks", 60, -3, 3);
  mGoodChi2 = std::make_unique<TH1D>("goodChi2", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", 200, 0, 100);

  mFakePt = std::make_unique<TH1D>("fakePt", ";#it{p}_{T} (GeV/#it{c});Fak", pars.effHistBins, xbins.data());
  mFakeEta = std::make_unique<TH1D>("fakeEta", ";#eta;Number of tracks", 60, -3, 3);
  mFakeChi2 = std::make_unique<TH1D>("fakeChi2", ";#it{p}_{T} (GeV/#it{c});Fak", 200, 0, 100);

  mMultiFake = std::make_unique<TH1D>("multiFake", ";#it{p}_{T} (GeV/#it{c});Fak", pars.effHistBins, xbins.data());

  mClonePt = std::make_unique<TH1D>("clonePt", ";#it{p}_{T} (GeV/#it{c});Clone", pars.effHistBins, xbins.data());
  mCloneEta = std::make_unique<TH1D>("cloneEta", ";#eta;Number of tracks", 60, -3, 3);

  mDenominatorPt = std::make_unique<TH1D>("denominatorPt", ";#it{p}_{T} (GeV/#it{c});Den", pars.effHistBins, xbins.data());
  mDenominatorEta = std::make_unique<TH1D>("denominatorEta", ";#eta;Number of tracks", 60, -3, 3);

  mGoodPt0 = std::make_unique<TH1D>("goodPt0", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
  mGoodPt1 = std::make_unique<TH1D>("goodPt1", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
  mGoodPt2 = std::make_unique<TH1D>("goodPt2", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());
  mGoodPt3 = std::make_unique<TH1D>("goodPt3", ";#it{p}_{T} (GeV/#it{c});Efficiency (fake-track rate)", pars.effHistBins, xbins.data());

  mFakePt0 = std::make_unique<TH1D>("FakePt0", ";#it{p}_{T} (GeV/#it{c});Efficiency (Fake-track rate)", pars.effHistBins, xbins.data());
  mFakePt1 = std::make_unique<TH1D>("FakePt1", ";#it{p}_{T} (GeV/#it{c});Efficiency (Fake-track rate)", pars.effHistBins, xbins.data());
  mFakePt2 = std::make_unique<TH1D>("FakePt2", ";#it{p}_{T} (GeV/#it{c});Efficiency (Fake-track rate)", pars.effHistBins, xbins.data());
  mFakePt3 = std::make_unique<TH1D>("FakePt3", ";#it{p}_{T} (GeV/#it{c});Efficiency (Fake-track rate)", pars.effHistBins, xbins.data());

  mGoodEta0 = std::make_unique<TH1D>("goodEta0", ";#eta;Number of tracks", 60, -3, 3);
  mGoodEta1 = std::make_unique<TH1D>("goodEta1", ";#eta;Number of tracks", 60, -3, 3);
  mGoodEta2 = std::make_unique<TH1D>("goodEta2", ";#eta;Number of tracks", 60, -3, 3);
  mGoodEta3 = std::make_unique<TH1D>("goodEta3", ";#eta;Number of tracks", 60, -3, 3);

  mFakeEta0 = std::make_unique<TH1D>("FakeEta0", ";#eta;Number of tracks", 60, -3, 3);
  mFakeEta1 = std::make_unique<TH1D>("FakeEta1", ";#eta;Number of tracks", 60, -3, 3);
  mFakeEta2 = std::make_unique<TH1D>("FakeEta2", ";#eta;Number of tracks", 60, -3, 3);
  mFakeEta3 = std::make_unique<TH1D>("FakeEta3", ";#eta;Number of tracks", 60, -3, 3);

  mPtSec0Pt = std::make_unique<TH1D>("mPtSec0Pt", ";#it{p}_{T} (GeV/#it{c}); ;#it{p}_{T} (GeV/#it{c})", pars.effHistBins, xbins.data());
  mPtSec1Pt = std::make_unique<TH1D>("mPtSec1Pt", ";#it{p}_{T} (GeV/#it{c}); ;#it{p}_{T} (GeV/#it{c})", pars.effHistBins, xbins.data());
  mPtSec2Pt = std::make_unique<TH1D>("mPtSec2Pt", ";#it{p}_{T} (GeV/#it{c}); ;#it{p}_{T} (GeV/#it{c})", pars.effHistBins, xbins.data());
  mPtSec3Pt = std::make_unique<TH1D>("mPtSec3Pt", ";#it{p}_{T} (GeV/#it{c}); ;#it{p}_{T} (GeV/#it{c})", pars.effHistBins, xbins.data());

  mPtSec0Eta = std::make_unique<TH1D>("mPtSec0Eta", ";#eta;Number of tracks", 60, -3, 3);
  mPtSec1Eta = std::make_unique<TH1D>("mPtSec1Eta", ";#eta;Number of tracks", 60, -3, 3);
  mPtSec2Eta = std::make_unique<TH1D>("mPtSec2Eta", ";#eta;Number of tracks", 60, -3, 3);
  mPtSec3Eta = std::make_unique<TH1D>("mPtSec3Eta", ";#eta;Number of tracks", 60, -3, 3);

  mGoodPt0->Sumw2();
  mGoodPt1->Sumw2();
  mGoodPt2->Sumw2();
  mGoodPt3->Sumw2();

  mFakePt0->Sumw2();
  mFakePt1->Sumw2();
  mFakePt2->Sumw2();
  mFakePt3->Sumw2();

  mGoodEta0->Sumw2();
  mGoodEta1->Sumw2();
  mGoodEta2->Sumw2();
  mGoodEta3->Sumw2();

  mFakeEta0->Sumw2();
  mFakeEta1->Sumw2();
  mFakeEta2->Sumw2();
  mFakeEta3->Sumw2();

  mPtSec0Pt->Sumw2();
  mPtSec1Pt->Sumw2();
  mPtSec2Pt->Sumw2();
  mPtSec3Pt->Sumw2();

  mPtSec0Eta->Sumw2();
  mPtSec1Eta->Sumw2();
  mPtSec2Eta->Sumw2();
  mPtSec3Eta->Sumw2();

  mGoodPt->Sumw2();
  mGoodEta->Sumw2();
  mFakePt->Sumw2();
  mMultiFake->Sumw2();
  mClonePt->Sumw2();
  mDenominatorPt->Sumw2();
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
  mParticleInfo.resize(mKineReader->getNSources());                                          // sources
  for (int iSource{0}; iSource < mKineReader->getNSources(); ++iSource) {
    mParticleInfo[iSource].resize(mKineReader->getNEvents(iSource));                         // events
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
  int good0 = 0, good1 = 0, good2 = 0, good3 = 0, totalsec = 0; // secondary
  int fake0 = 0, fake1 = 0, fake2 = 0, fake3 = 0;
  int totsec0 = 0, totsec1 = 0, totsec2 = 0, totsec3 = 0;
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

  // Currently process only sourceID = 0, to be extended later if needed
  for (auto& evInfo : mParticleInfo[0]) {
    for (auto& part : evInfo) {
      if ((part.clusters & 0x7f) != mMask) {
        // part.clusters != 0x3f && part.clusters != 0x3f << 1 &&
        // part.clusters != 0x1f && part.clusters != 0x1f << 1 && part.clusters != 0x1f << 2 &&
        // part.clusters != 0x0f && part.clusters != 0x0f << 1 && part.clusters != 0x0f << 2 && part.clusters != 0x0f << 3) {
        continue;
      }
      if (part.isPrimary) {
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

      if (!part.isPrimary) {
        totalsec++;
        float rad = sqrt(pow(part.vx, 2) + pow(part.vy, 2));
        if (rad < rLayer0) // layer 0
        {
          totsec0++;
          mPtSec0Pt->Fill(part.pt);
          mPtSec0Eta->Fill(part.eta);

          if (part.isReco) {
            mGoodPt0->Fill(part.pt);
            mGoodEta0->Fill(part.eta);
            good0++;
          }
          if (part.isFake) {
            mFakePt0->Fill(part.pt);
            mFakeEta0->Fill(part.eta);
            fake0++;
          }
        }

        if (rad < rLayer1 && rad > rLayer0) // layer 1
        {
          totsec1++;
          mPtSec1Pt->Fill(part.pt);
          mPtSec1Eta->Fill(part.eta);
          if (part.isReco) {
            mGoodPt1->Fill(part.pt);
            mGoodEta1->Fill(part.eta);
            good1++;
          }
          if (part.isFake) {
            mFakePt1->Fill(part.pt);
            mFakeEta1->Fill(part.eta);
            fake1++;
          }
        }

        if (rad < rLayer2 && rad > rLayer1) // layer 2
        {
          totsec2++;
          mPtSec2Pt->Fill(part.pt);
          mPtSec2Eta->Fill(part.eta);
          if (part.isReco) {
            mGoodPt2->Fill(part.pt);
            mGoodEta2->Fill(part.eta);
            good2++;
          }
          if (part.isFake) {
            mFakePt2->Fill(part.pt);
            mFakeEta2->Fill(part.eta);
            fake2++;
          }
        }

        if (rad < rLayer3 && rad > rLayer2) // layer 3
        {
          totsec3++;
          mPtSec3Pt->Fill(part.pt);
          mPtSec3Eta->Fill(part.eta);
          if (part.isReco) {
            mGoodPt3->Fill(part.pt);
            mGoodEta3->Fill(part.eta);
            good3++;
          }
          if (part.isFake) {
            mFakePt3->Fill(part.pt);
            mFakeEta3->Fill(part.eta);
            fake3++;
          }
        }
      }
    }
  }
  LOGP(info, "** Some statistics on secondary tracks:");

  LOGP(info, "\t- Total number of secondary tracks: {}", totalsec);
  LOGP(info, "\t- Total number of secondary tracks on layer O: {}, good: {}, fake: {}", totsec0, good0, fake0);
  LOGP(info, "\t- Total number of secondary tracks on layer 1: {}, good: {}, fake: {}", totsec1, good1, fake1);
  LOGP(info, "\t- Total number of secondary tracks on layer 2: {}, good: {}, fake: {}", totsec2, good2, fake2);
  LOGP(info, "\t- Total number of secondary tracks on layer 3: {}, good: {}, fake: {}", totsec3, good3, fake3);

  LOGP(info, "** Computing efficiencies ...");

  mEffPt = std::make_unique<TEfficiency>(*mGoodPt, *mDenominatorPt);
  mEffFakePt = std::make_unique<TEfficiency>(*mFakePt, *mDenominatorPt);
  mEffClonesPt = std::make_unique<TEfficiency>(*mClonePt, *mDenominatorPt);

  mEffEta = std::make_unique<TEfficiency>(*mGoodEta, *mDenominatorEta);
  mEffFakeEta = std::make_unique<TEfficiency>(*mFakeEta, *mDenominatorEta);
  mEffClonesEta = std::make_unique<TEfficiency>(*mCloneEta, *mDenominatorEta);

  mEff0Pt = std::make_unique<TEfficiency>(*mGoodPt0, *mPtSec0Pt);
  mEff1Pt = std::make_unique<TEfficiency>(*mGoodPt1, *mPtSec1Pt);
  mEff2Pt = std::make_unique<TEfficiency>(*mGoodPt2, *mPtSec2Pt);
  mEff3Pt = std::make_unique<TEfficiency>(*mGoodPt3, *mPtSec3Pt);

  mEff0FakePt = std::make_unique<TEfficiency>(*mFakePt0, *mPtSec0Pt);
  mEff1FakePt = std::make_unique<TEfficiency>(*mFakePt1, *mPtSec1Pt);
  mEff2FakePt = std::make_unique<TEfficiency>(*mFakePt2, *mPtSec2Pt);
  mEff3FakePt = std::make_unique<TEfficiency>(*mFakePt3, *mPtSec3Pt);

  mEff0Eta = std::make_unique<TEfficiency>(*mGoodEta0, *mPtSec0Eta);
  mEff1Eta = std::make_unique<TEfficiency>(*mGoodEta1, *mPtSec1Eta);
  mEff2Eta = std::make_unique<TEfficiency>(*mGoodEta2, *mPtSec2Eta);
  mEff3Eta = std::make_unique<TEfficiency>(*mGoodEta3, *mPtSec3Eta);

  mEff0FakeEta = std::make_unique<TEfficiency>(*mFakeEta0, *mPtSec0Eta);
  mEff1FakeEta = std::make_unique<TEfficiency>(*mFakeEta1, *mPtSec1Eta);
  mEff2FakeEta = std::make_unique<TEfficiency>(*mFakeEta2, *mPtSec2Eta);
  mEff3FakeEta = std::make_unique<TEfficiency>(*mFakeEta3, *mPtSec3Eta);
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
  mEffPt->SetName("Good_pt");
  mEffPt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency primary particle");
  mEffPt->SetLineColor(kAzure + 4);
  mEffPt->SetLineColorAlpha(kAzure + 4, 0.65);
  mEffPt->SetLineWidth(2);
  mEffPt->SetMarkerColorAlpha(kAzure + 4, 0.65);
  mEffPt->SetMarkerStyle(kFullCircle);
  mEffPt->SetMarkerSize(1.35);
  mEffPt->SetDirectory(gDirectory);
  fout.WriteTObject(mEffPt.get());

  mEffFakePt->SetName("Fake_pt");
  mEffFakePt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency primary particle");
  mEffFakePt->SetLineColor(kRed + 1);
  mEffFakePt->SetLineColorAlpha(kRed + 1, 0.65);
  mEffFakePt->SetLineWidth(2);
  mEffFakePt->SetMarkerColorAlpha(kRed + 1, 0.65);
  mEffFakePt->SetMarkerStyle(kFullCircle);
  mEffFakePt->SetMarkerSize(1.35);
  mEffFakePt->SetDirectory(gDirectory);
  fout.WriteTObject(mEffFakePt.get());

  mEffClonesPt->SetName("Clone_pt");
  mEffClonesPt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency primary particle");
  mEffClonesPt->SetLineColor(kGreen + 2);
  mEffClonesPt->SetLineColorAlpha(kGreen + 2, 0.65);
  mEffClonesPt->SetLineWidth(2);
  mEffClonesPt->SetMarkerColorAlpha(kGreen + 2, 0.65);

  mEffClonesPt->SetMarkerStyle(kFullCircle);
  mEffClonesPt->SetMarkerSize(1.35);
  mEffClonesPt->SetDirectory(gDirectory);
  fout.WriteTObject(mEffClonesPt.get());

  mEffEta->SetName("Good_eta");
  mEffEta->SetTitle(";#eta;efficiency primary particle");
  mEffEta->SetLineColor(kAzure + 4);
  mEffEta->SetLineColorAlpha(kAzure + 4, 0.65);
  mEffEta->SetLineWidth(2);
  mEffEta->SetMarkerColorAlpha(kAzure + 4, 0.65);
  mEffEta->SetMarkerStyle(kFullCircle);
  mEffEta->SetMarkerSize(1.35);
  mEffEta->SetDirectory(gDirectory);
  fout.WriteTObject(mEffEta.get());

  mEffFakeEta->SetName("Fake_eta");
  mEffFakeEta->SetTitle(";#eta;efficiency primary particle");
  mEffFakeEta->SetLineColor(kRed + 1);
  mEffFakeEta->SetLineColorAlpha(kRed + 1, 0.65);
  mEffFakeEta->SetLineWidth(2);
  mEffFakeEta->SetMarkerColorAlpha(kRed + 1, 0.65);
  mEffFakeEta->SetMarkerStyle(kFullCircle);
  mEffFakeEta->SetMarkerSize(1.35);
  mEffFakeEta->SetDirectory(gDirectory);
  fout.WriteTObject(mEffFakeEta.get());

  mEffClonesEta->SetName("Clone_eta");
  mEffClonesEta->SetTitle(";#eta;efficiency primary particle");
  mEffClonesEta->SetLineColor(kGreen + 2);
  mEffClonesEta->SetLineColorAlpha(kGreen + 2, 0.65);
  mEffClonesEta->SetLineWidth(2);
  mEffClonesEta->SetMarkerColorAlpha(kGreen + 2, 0.65);
  mEffClonesEta->SetMarkerStyle(kFullCircle);
  mEffClonesEta->SetMarkerSize(1.35);
  mEffClonesEta->SetDirectory(gDirectory);
  fout.WriteTObject(mEffClonesEta.get());

  mEff0Pt->SetName("Good_pt0"); //******LAYER 0******
  mEff0Pt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff0Pt->SetLineColor(kAzure + 4);
  mEff0Pt->SetLineColorAlpha(kAzure + 4, 1);
  mEff0Pt->SetLineWidth(2);
  mEff0Pt->SetMarkerColorAlpha(kAzure + 4, 1);
  mEff0Pt->SetMarkerStyle(kFullCircle);
  mEff0Pt->SetMarkerSize(1.7);
  mEff0Pt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff0Pt.get());

  mEff0FakePt->SetName("Fake_pt0");
  mEff0FakePt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff0FakePt->SetLineColor(kAzure + 4);
  mEff0FakePt->SetLineColorAlpha(kAzure + 4, 1);
  mEff0FakePt->SetLineWidth(2);
  mEff0FakePt->SetMarkerColorAlpha(kAzure + 4, 1);
  mEff0FakePt->SetMarkerStyle(kFullCircle);
  mEff0FakePt->SetMarkerSize(1.7);
  mEff0FakePt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff0FakePt.get());

  mEff0Eta->SetName("Good_eta0");
  mEff0Eta->SetTitle(";#eta;efficiency secondary particle");
  mEff0Eta->SetLineColor(kAzure + 4);
  mEff0Eta->SetLineColorAlpha(kAzure + 4, 1);
  mEff0Eta->SetLineWidth(2);
  mEff0Eta->SetMarkerColorAlpha(kAzure + 4, 1);
  mEff0Eta->SetMarkerStyle(kFullCircle);
  mEff0Eta->SetMarkerSize(1.7);
  mEff0Eta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff0Eta.get());

  mEff0FakeEta->SetName("Fake_eta0");
  mEff0FakeEta->SetTitle(";#eta;efficiency secondary particle");
  mEff0FakeEta->SetLineColor(kAzure + 4);
  mEff0FakeEta->SetLineColorAlpha(kAzure + 4, 1);
  mEff0FakeEta->SetLineWidth(2);
  mEff0FakeEta->SetMarkerColorAlpha(kAzure + 4, 1);
  mEff0FakeEta->SetMarkerStyle(kFullCircle);
  mEff0FakeEta->SetMarkerSize(1.7);
  mEff0FakeEta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff0FakeEta.get());

  mEff1Pt->SetName("Good_pt1"); //*****LAYER 1 ********
  mEff1Pt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff1Pt->SetLineColor(kRed);
  mEff1Pt->SetLineColorAlpha(kRed, 1);
  mEff1Pt->SetLineWidth(2);
  mEff1Pt->SetMarkerColorAlpha(kRed, 1);
  mEff1Pt->SetMarkerStyle(kFullCircle);
  mEff1Pt->SetMarkerSize(1.7);
  mEff1Pt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff1Pt.get());

  mEff1FakePt->SetName("Fake_pt1");
  mEff1FakePt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff1FakePt->SetLineColor(kRed);
  mEff1FakePt->SetLineColorAlpha(kRed, 1);
  mEff1FakePt->SetLineWidth(2);
  mEff1FakePt->SetMarkerColorAlpha(kRed, 1);
  mEff1FakePt->SetMarkerStyle(kFullCircle);
  mEff1FakePt->SetMarkerSize(1.7);
  mEff1FakePt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff1FakePt.get());

  mEff1Eta->SetName("Good_eta1");
  mEff1Eta->SetTitle(";#eta;efficiency secondary particle");
  mEff1Eta->SetLineColor(kRed);
  mEff1Eta->SetLineColorAlpha(kRed, 1);
  mEff1Eta->SetLineWidth(2);
  mEff1Eta->SetMarkerColorAlpha(kRed, 1);
  mEff1Eta->SetMarkerStyle(kFullCircle);
  mEff1Eta->SetMarkerSize(1.7);
  mEff1Eta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff1Eta.get());

  mEff1FakeEta->SetName("Fake_eta1");
  mEff1FakeEta->SetTitle(";#eta;efficiency secondary particle");
  mEff1FakeEta->SetLineColor(kRed);
  mEff1FakeEta->SetLineColorAlpha(kRed, 1);
  mEff1FakeEta->SetLineWidth(2);
  mEff1FakeEta->SetMarkerColorAlpha(kRed, 1);
  mEff1FakeEta->SetMarkerStyle(kFullCircle);
  mEff1FakeEta->SetMarkerSize(1.7);
  mEff1FakeEta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff1FakeEta.get());

  mEff2Pt->SetName("Good_pt2"); //*****LAYER 2 ********
  mEff2Pt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff2Pt->SetLineColor(kGreen + 1);
  mEff2Pt->SetLineColorAlpha(kGreen + 1, 1);
  mEff2Pt->SetLineWidth(2);
  mEff2Pt->SetMarkerColorAlpha(kGreen + 1, 1);
  mEff2Pt->SetMarkerStyle(kFullCircle);
  mEff2Pt->SetMarkerSize(1.7);
  mEff2Pt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff2Pt.get());

  mEff2FakePt->SetName("Fake_pt2");
  mEff2FakePt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff2FakePt->SetLineColor(kGreen + 1);
  mEff2FakePt->SetLineColorAlpha(kGreen + 1, 1);
  mEff2FakePt->SetLineWidth(2);
  mEff2FakePt->SetMarkerColorAlpha(kGreen + 1, 1);
  mEff2FakePt->SetMarkerStyle(kFullCircle);
  mEff2FakePt->SetMarkerSize(1.7);
  mEff2FakePt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff2FakePt.get());

  mEff2Eta->SetName("Good_eta2");
  mEff2Eta->SetTitle(";#eta;efficiency secondary particle");
  mEff2Eta->SetLineColor(kGreen + 1);
  mEff2Eta->SetLineColorAlpha(kGreen + 1, 1);
  mEff2Eta->SetLineWidth(2);
  mEff2Eta->SetMarkerColorAlpha(kGreen + 1, 1);
  mEff2Eta->SetMarkerStyle(kFullCircle);
  mEff2Eta->SetMarkerSize(1.7);
  mEff2Eta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff2Eta.get());

  mEff2FakeEta->SetName("Fake_eta2");
  mEff2FakeEta->SetTitle(";#eta;efficiency secondary particle");
  mEff2FakeEta->SetLineColor(kGreen + 1);
  mEff2FakeEta->SetLineColorAlpha(kGreen + 1, 1);
  mEff2FakeEta->SetLineWidth(2);
  mEff2FakeEta->SetMarkerColorAlpha(kGreen + 1, 1);
  mEff2FakeEta->SetMarkerStyle(kFullCircle);
  mEff2FakeEta->SetMarkerSize(1.7);
  mEff2FakeEta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff2FakeEta.get());

  mEff3Pt->SetName("Good_pt3"); //*****LAYER 3 ********
  mEff3Pt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff3Pt->SetLineColor(kOrange - 3);
  mEff3Pt->SetLineColorAlpha(kOrange - 3, 1);
  mEff3Pt->SetLineWidth(2);
  mEff3Pt->SetMarkerColorAlpha(kOrange - 3, 1);
  mEff3Pt->SetMarkerStyle(kFullCircle);
  mEff3Pt->SetMarkerSize(1.7);
  mEff3Pt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff3Pt.get());

  mEff3FakePt->SetName("Fake_pt3");
  mEff3FakePt->SetTitle(";#it{p}_{T} (GeV/#it{c});efficiency secondary particle ");
  mEff3FakePt->SetLineColor(kOrange - 3);
  mEff3FakePt->SetLineColorAlpha(kOrange - 3, 1);
  mEff3FakePt->SetLineWidth(2);
  mEff3FakePt->SetMarkerColorAlpha(kOrange - 3, 1);
  mEff3FakePt->SetMarkerStyle(kFullCircle);
  mEff3FakePt->SetMarkerSize(1.7);
  mEff3FakePt->SetDirectory(gDirectory);
  fout.WriteTObject(mEff3FakePt.get());

  mEff3Eta->SetName("Good_eta3");
  mEff3Eta->SetTitle(";#eta;efficiency secondary particle");
  mEff3Eta->SetLineColor(kOrange - 3);
  mEff3Eta->SetLineColorAlpha(kOrange - 3, 1);
  mEff3Eta->SetLineWidth(2);
  mEff3Eta->SetMarkerColorAlpha(kOrange - 3, 1);
  mEff3Eta->SetMarkerStyle(kFullCircle);
  mEff3Eta->SetMarkerSize(1.7);
  mEff3Eta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff3Eta.get());

  mEff3FakeEta->SetName("Fake_eta3");
  mEff3FakeEta->SetTitle(";#eta;efficiency secondary particle");
  mEff3FakeEta->SetLineColor(kOrange - 3);
  mEff3FakeEta->SetLineColorAlpha(kOrange - 3, 1);
  mEff3FakeEta->SetLineWidth(2);
  mEff3FakeEta->SetMarkerColorAlpha(kOrange - 3, 1);
  mEff3FakeEta->SetMarkerStyle(kFullCircle);
  mEff3FakeEta->SetMarkerSize(1.7);
  mEff3FakeEta->SetDirectory(gDirectory);
  fout.WriteTObject(mEff3FakeEta.get());

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

  mCanvasPt2 = std::make_unique<TCanvas>("cPt2", "cPt2", 1600, 1200);
  mCanvasPt2->cd();
  mCanvasPt2->SetLogx();
  mCanvasPt2->SetGrid();
  mEff0Pt->Draw("pz");
  mEff1Pt->Draw("pz same");
  mEff2Pt->Draw("pz same");
  mEff3Pt->Draw("pz same");

  mLegendPt2 = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendPt2->SetHeader(Form("%zu events PP, good tracks", mKineReader->getNEvents(0)), "C");
  mLegendPt2->AddEntry("Good_pt0", "Layer 0", "lep");
  mLegendPt2->AddEntry("Good_pt1", "Layer 1", "lep");
  mLegendPt2->AddEntry("Good_pt2", "Layer 2", "lep");
  mLegendPt2->AddEntry("Good_pt3", "Layer 3", "lep");

  mLegendPt2->Draw();
  mCanvasPt2->SaveAs("eff_sec_pt.png");

  mCanvasEta2 = std::make_unique<TCanvas>("cEta2", "cEta2", 1600, 1200);
  mCanvasEta2->cd();

  mCanvasEta2->SetGrid();
  mEff0Eta->Draw("pz");
  mEff1Eta->Draw("pz same");
  mEff2Eta->Draw("pz same");
  mEff3Eta->Draw("pz same");

  mLegendEta2 = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendEta2->SetHeader(Form("%zu events PP, good tracks", mKineReader->getNEvents(0)), "C");
  mLegendEta2->AddEntry("Good_eta0", "Layer 0", "lep");
  mLegendEta2->AddEntry("Good_eta1", "Layer 1", "lep");
  mLegendEta2->AddEntry("Good_eta2", "Layer 2", "lep");
  mLegendEta2->AddEntry("Good_eta3", "Layer 3", "lep");

  mLegendEta2->Draw();
  mCanvasEta2->SaveAs("eff_sec_eta.png");

  mCanvasPt2fake = std::make_unique<TCanvas>("cPt2fake", "cPt2fake", 1600, 1200);
  mCanvasPt2fake->cd();
  mCanvasPt2fake->SetLogx();
  mCanvasPt2fake->SetGrid();
  mEff0FakePt->Draw("pz");
  mEff1FakePt->Draw("pz same");
  mEff2FakePt->Draw("pz same");
  mEff3FakePt->Draw("pz same");

  mLegendPt2Fake = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendPt2Fake->SetHeader(Form("%zu events PP, fake tracks ", mKineReader->getNEvents(0)), "C");
  mLegendPt2Fake->AddEntry("Fake_pt0", "Layer 0", "lep");
  mLegendPt2Fake->AddEntry("Fake_pt1", "Layer 1", "lep");
  mLegendPt2Fake->AddEntry("Fake_pt2", "Layer 2", "lep");
  mLegendPt2Fake->AddEntry("Fake_pt3", "Layer 3", "lep");

  mLegendPt2Fake->Draw();
  mCanvasPt2fake->SaveAs("eff_sec_pt_fake.png");

  mCanvasEta2fake = std::make_unique<TCanvas>("cEta2fake", "cEta2fake", 1600, 1200);
  mCanvasEta2fake->cd();

  mCanvasEta2fake->SetGrid();
  mEff0FakeEta->Draw("pz");
  mEff1FakeEta->Draw("pz same");
  mEff2FakeEta->Draw("pz same");
  mEff3FakeEta->Draw("pz same");

  mLegendEta2Fake = std::make_unique<TLegend>(0.19, 0.8, 0.40, 0.96);
  mLegendEta2Fake->SetHeader(Form("%zu events PP, fake tracks ", mKineReader->getNEvents(0)), "C");
  mLegendEta2Fake->AddEntry("Fake_eta0", "Layer 0", "lep");
  mLegendEta2Fake->AddEntry("Fake_eta1", "Layer 1", "lep");
  mLegendEta2Fake->AddEntry("Fake_eta2", "Layer 2", "lep");
  mLegendEta2Fake->AddEntry("Fake_eta3", "Layer 3", "lep");

  mLegendEta2Fake->Draw();
  mCanvasEta2fake->SaveAs("eff_sec_Eta_fake.png");

  fout.cd();
  mCanvasPt->Write();
  mCanvasEta->Write();
  mCanvasPt2->Write();
  mCanvasPt2fake->Write();
  mCanvasEta2->Write();
  mCanvasEta2fake->Write();

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