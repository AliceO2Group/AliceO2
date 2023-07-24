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

  // Canvas & decorations
  std::unique_ptr<TCanvas> mCanvasPt;
  std::unique_ptr<TCanvas> mCanvasEta;
  std::unique_ptr<TLegend> mLegendPt;
  std::unique_ptr<TLegend> mLegendEta;

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
        mParticleInfo[iSource][iEvent][iPart].isPrimary = part.isPrimary();
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
      if (!part.isPrimary) {
        continue;
      }
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
  LOGP(info, "** Computing efficiencies ...");

  mEffPt = std::make_unique<TEfficiency>(*mGoodPt, *mDenominatorPt);
  mEffFakePt = std::make_unique<TEfficiency>(*mFakePt, *mDenominatorPt);
  mEffClonesPt = std::make_unique<TEfficiency>(*mClonePt, *mDenominatorPt);

  mEffEta = std::make_unique<TEfficiency>(*mGoodEta, *mDenominatorEta);
  mEffFakeEta = std::make_unique<TEfficiency>(*mFakeEta, *mDenominatorEta);
  mEffClonesEta = std::make_unique<TEfficiency>(*mCloneEta, *mDenominatorEta);
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

  fout.cd();
  mCanvasPt->Write();
  mCanvasEta->Write();
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