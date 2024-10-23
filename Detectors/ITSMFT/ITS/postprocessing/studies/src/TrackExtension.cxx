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

#include "CommonUtils/TreeStreamRedirector.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSStudies/Helpers.h"
#include "ITSStudies/TrackExtension.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"
#include "Steer/MCKinematicsReader.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "ReconstructionDataFormats/DCA.h"

#include <bitset>

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TEfficiency.h"

namespace o2::its::study
{
using namespace o2::framework;
using namespace o2::globaltracking;

using GTrackID = o2::dataformats::GlobalTrackID;
using o2::steer::MCKinematicsReader;
class TrackExtensionStudy : public Task
{
  struct ParticleInfo {
    float eventX;
    float eventY;
    float eventZ;
    int pdg;
    float pt;
    float eta;
    float phi;
    int mother;
    int first;
    float vx;
    float vy;
    float vz;
    uint8_t clusters = 0u;
    uint8_t fakeClusters = 0u;
    uint8_t isReco = 0u;
    uint8_t isFake = 0u;
    bool isPrimary = false;
    unsigned char storedStatus = 2; /// not stored = 2, fake = 1, good = 0
    int prodProcess;
    o2::its::TrackITS track;
    MCTrack mcTrack;
  };

 public:
  TrackExtensionStudy(std::shared_ptr<DataRequest> dr,
                      mask_t src,
                      std::shared_ptr<o2::steer::MCKinematicsReader> kineReader,
                      std::shared_ptr<o2::base::GRPGeomRequest> gr) : mDataRequest(dr), mTracksSrc(src), mKineReader(kineReader), mGGCCDBRequest(gr)
  {
    LOGP(info, "Read MCKine reader with {} sources", mKineReader->getNSources());
  }

  ~TrackExtensionStudy() final = default;
  void init(InitContext& /*ic*/) final;
  void run(ProcessingContext& /*pc*/) final;
  void endOfStream(EndOfStreamContext& /*ec*/) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process();

 private:
  static constexpr std::array<uint8_t, 9> mBitPatternsBefore{15, 30, 31, 60, 62, 63, 120, 124, 126};
  static constexpr std::array<uint8_t, 16> mBitPatternsAfter{31, 47, 61, 62, 63, 79, 94, 95, 111, 121, 122, 123, 124, 125, 126, 127};
  const std::bitset<7> mTopMask{"1110000"};
  const std::bitset<7> mBotMask{"0000111"};

  void updateTimeDependentParams(ProcessingContext& pc);
  std::string mOutFileName = "TrackExtensionStudy.root";
  std::shared_ptr<MCKinematicsReader> mKineReader;
  GeometryTGeo* mGeometry{};

  gsl::span<const o2::itsmft::ROFRecord> mTracksROFRecords;
  gsl::span<const o2::its::TrackITS> mTracks;
  gsl::span<const o2::MCCompLabel> mTracksMCLabels;
  gsl::span<const o2::itsmft::CompClusterExt> mClusters;
  gsl::span<const int> mInputITSidxs;
  const o2::dataformats::MCLabelContainer* mClustersMCLCont{};

  GTrackID::mask_t mTracksSrc{};
  std::shared_ptr<DataRequest> mDataRequest;
  std::vector<std::vector<std::vector<ParticleInfo>>> mParticleInfo; // src/event/track
  unsigned short mMask = 0x7f;

  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<utils::TreeStreamRedirector> mStream;
  bool mWithTree{false};

  std::unique_ptr<TH1D> mHTrackCounts;
  std::unique_ptr<TH1D> mHLengthAny, mHLengthGood, mHLengthFake;
  std::unique_ptr<TH1D> mHChi2Any, mHChi2Good, mHChi2Fake;
  std::unique_ptr<TH1D> mHPtAny, mHPtGood, mHPtFake;
  std::unique_ptr<TH1D> mHExtensionAny, mHExtensionGood, mHExtensionFake;
  std::unique_ptr<TH2D> mHExtensionPatternsAny, mHExtensionPatternsGood, mHExtensionPatternsFake, mHExtensionPatternsGoodMissed, mHExtensionPatternsGoodEmpty;
  std::unique_ptr<TH1D> mEExtensionNum, mEExtensionDen, mEExtensionPurityNum, mEExtensionPurityDen, mEExtensionFakeNum, mEExtensionFakeDen;
  std::unique_ptr<TH1D> mEExtensionFakeBeforeNum, mEExtensionFakeAfterNum, mEExtensionFakeMixNum;
  std::unique_ptr<TH1D> mEExtensionTopNum, mEExtensionTopPurityNum, mEExtensionTopFakeNum;
  std::unique_ptr<TH1D> mEExtensionBotNum, mEExtensionBotPurityNum, mEExtensionBotFakeNum;
  std::unique_ptr<TH1D> mEExtensionMixNum, mEExtensionMixPurityNum, mEExtensionMixFakeNum;
  std::array<std::unique_ptr<TH1D>, mBitPatternsBefore.size()> mEExtensionPatternGoodNum, mEExtensionPatternFakeNum;
  std::array<std::array<std::unique_ptr<TH1D>, mBitPatternsAfter.size()>, mBitPatternsBefore.size()> mEExtensionPatternIndGoodNum, mEExtensionPatternIndFakeNum;
  // DCA
  std::unique_ptr<TH2D> mDCAxyVsPtPionsNormal, mDCAxyVsPtPionsExtended;
  std::unique_ptr<TH2D> mDCAzVsPtPionsNormal, mDCAzVsPtPionsExtended;

  template <class T, typename... C, typename... F>
  std::unique_ptr<T> createHistogram(C... n, F... b)
  {
    auto t = std::make_unique<T>(n..., b...);
    mHistograms.push_back(static_cast<TH1*>(t.get()));
    return std::move(t);
  }
  std::vector<TH1*> mHistograms;
};

void TrackExtensionStudy::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mWithTree = ic.options().get<bool>("with-tree");

  constexpr size_t effHistBins = 40;
  constexpr float effPtCutLow = 0.01;
  constexpr float effPtCutHigh = 10.;
  auto xbins = helpers::makeLogBinning(effHistBins, effPtCutLow, effPtCutHigh);

  // Track Counting
  mHTrackCounts = createHistogram<TH1D>("hTrackCounts", "Track Stats", 10, 0, 10);
  mHTrackCounts->GetXaxis()->SetBinLabel(1, "Total Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(2, "Normal ANY Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(3, "Normal GOOD Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(4, "Normal FAKE Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(5, "Extended ANY Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(6, "Extended GOOD Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(7, "Extended FAKE Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(8, "Extended FAKE BEFORE Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(9, "Extended FAKE AFTER Tracks");
  mHTrackCounts->GetXaxis()->SetBinLabel(10, "Extended FAKE BEFORE&AFTER Tracks");

  // Length
  mHLengthAny = createHistogram<TH1D>("hLengthAny", "Extended Tracks Length (ANY);NCluster;Entries", 5, 3, 8);
  mHLengthGood = createHistogram<TH1D>("hLengthGood", "Extended Tracks Length (GOOD);NCluster;Entries", 5, 3, 8);
  mHLengthFake = createHistogram<TH1D>("hLengthFake", "Extended Tracks Length (FAKE);NCluster;Entries", 5, 3, 8);

  // Chi2
  mHChi2Any = createHistogram<TH1D>("hChi2Any", "Extended Tracks Length (ANY);#chi^{2};Entries", 50, 0, 100);
  mHChi2Good = createHistogram<TH1D>("hChi2Good", "Extended Tracks Length (GOOD);#chi^{2};Entries", 50, 0, 100);
  mHChi2Fake = createHistogram<TH1D>("hChi2Fake", "Extended Tracks Length (FAKE);#chi^{2};Entries", 50, 0, 100);

  // Pt
  mHPtAny = createHistogram<TH1D>("hPtAny", "Extended Tracks Length (ANY);#it{p}_{T};Entries", effHistBins, xbins.data());
  mHPtGood = createHistogram<TH1D>("hPtGood", "Extended Tracks Length (GOOD);#it{p}_{T};Entries", effHistBins, xbins.data());
  mHPtFake = createHistogram<TH1D>("hPtFake", "Extended Tracks Length (FAKE);#it{p}_{T};Entries", effHistBins, xbins.data());

  // Length
  mHExtensionAny = createHistogram<TH1D>("hExtensionAny", "Extended Tracks Length (ANY);Extended Layer;Entries", 7, 0, 7);
  mHExtensionGood = createHistogram<TH1D>("hExtensionGood", "Extended Tracks Length (GOOD);Extended Layer;Entries", 7, 0, 7);
  mHExtensionFake = createHistogram<TH1D>("hExtensionFake", "Extended Tracks Length (FAKE);Extended Layer;Entries", 7, 0, 7);

  // Patterns
  auto makePatternAxisLabels = [&](TH1* h, bool xBefore = true) {
    for (int i{1}; i <= h->GetXaxis()->GetNbins(); ++i) {
      if (xBefore) {
        h->GetXaxis()->SetBinLabel(i, fmt::format("{:07b}", mBitPatternsBefore[i - 1]).c_str());
      } else {
        h->GetXaxis()->SetBinLabel(i, fmt::format("{:07b}", mBitPatternsAfter[i - 1]).c_str());
      }
    }
    for (int i{1}; i <= h->GetYaxis()->GetNbins(); ++i) {
      h->GetYaxis()->SetBinLabel(i, fmt::format("{:07b}", mBitPatternsAfter[i - 1]).c_str());
    }
  };
  mHExtensionPatternsAny = createHistogram<TH2D>("hExtensionPatternsAny", "Extended Tracks Pattern (ANY);Before;After;Entries", mBitPatternsBefore.size(), 0, mBitPatternsBefore.size(), mBitPatternsAfter.size(), 0, mBitPatternsAfter.size());
  makePatternAxisLabels(mHExtensionPatternsAny.get());
  mHExtensionPatternsGood = createHistogram<TH2D>("hExtensionPatternsGood", "Extended Tracks Pattern (GOOD);Before;After;Entries", mBitPatternsBefore.size(), 0, mBitPatternsBefore.size(), mBitPatternsAfter.size(), 0, mBitPatternsAfter.size());
  makePatternAxisLabels(mHExtensionPatternsGood.get());
  mHExtensionPatternsFake = createHistogram<TH2D>("hExtensionPatternsFake", "Extended Tracks Pattern (FAKE);Before;After;Entries", mBitPatternsBefore.size(), 0, mBitPatternsBefore.size(), mBitPatternsAfter.size(), 0, mBitPatternsAfter.size());
  makePatternAxisLabels(mHExtensionPatternsFake.get());
  mHExtensionPatternsGoodMissed = createHistogram<TH2D>("hExtensionPatternsGoodMissed", "Extended Tracks Pattern (GOOD) Missed Clusters;After;Missed;Entries", mBitPatternsAfter.size(), 0, mBitPatternsAfter.size(), mBitPatternsAfter.size(), 0, mBitPatternsAfter.size());
  makePatternAxisLabels(mHExtensionPatternsGoodMissed.get(), false);
  mHExtensionPatternsGoodEmpty = createHistogram<TH2D>("hExtensionPatternsGoodEmpty", "Extended Tracks Pattern (GOOD) Empty Clusters;Before;After;Entries", mBitPatternsAfter.size(), 0, mBitPatternsAfter.size(), mBitPatternsAfter.size(), 0, mBitPatternsAfter.size());
  makePatternAxisLabels(mHExtensionPatternsGoodEmpty.get(), false);

  /// Effiencies
  mEExtensionNum = createHistogram<TH1D>("hExtensionNum", "Extension Numerator", effHistBins, xbins.data());
  mEExtensionDen = createHistogram<TH1D>("hExtensionDen", "Extension Dennominator", effHistBins, xbins.data());
  // Purity
  mEExtensionPurityNum = createHistogram<TH1D>("hExtensionPurityNum", "Extension Purity Numerator", effHistBins, xbins.data());
  mEExtensionPurityDen = createHistogram<TH1D>("hExtensionPurityDen", "Extension Purity Denominator", effHistBins, xbins.data());
  // Fake
  mEExtensionFakeNum = createHistogram<TH1D>("hExtensionFakeNum", "Extension Fake Numerator", effHistBins, xbins.data());
  mEExtensionFakeDen = createHistogram<TH1D>("hExtensionFakeDen", "Extension Fake Denominator", effHistBins, xbins.data());
  mEExtensionFakeBeforeNum = createHistogram<TH1D>("hExtensionFakeBeforeNum", "Extension Fake Before Numerator", effHistBins, xbins.data());
  mEExtensionFakeAfterNum = createHistogram<TH1D>("hExtensionFakeAfterNum", "Extension Fake After Numerator", effHistBins, xbins.data());
  mEExtensionFakeMixNum = createHistogram<TH1D>("hExtensionFakeMixNum", "Extension Fake Mix Numerator", effHistBins, xbins.data());
  // Top
  mEExtensionTopNum = createHistogram<TH1D>("hExtensionTopNum", "Extension Top Numerator", effHistBins, xbins.data());
  mEExtensionTopPurityNum = createHistogram<TH1D>("hExtensionTopPurityNum", "Extension Top Purity Numerator", effHistBins, xbins.data());
  mEExtensionTopFakeNum = createHistogram<TH1D>("hExtensionTopFakeNum", "Extension Top Fake Numerator", effHistBins, xbins.data());
  mEExtensionBotNum = createHistogram<TH1D>("hExtensionBotNum", "Extension Bot Numerator", effHistBins, xbins.data());
  mEExtensionBotPurityNum = createHistogram<TH1D>("hExtensionBotPurityNum", "Extension Bot Purity Numerator", effHistBins, xbins.data());
  mEExtensionBotFakeNum = createHistogram<TH1D>("hExtensionBotFakeNum", "Extension Bot Fake Numerator", effHistBins, xbins.data());
  mEExtensionMixNum = createHistogram<TH1D>("hExtensionMixNum", "Extension Mix Numerator", effHistBins, xbins.data());
  mEExtensionMixPurityNum = createHistogram<TH1D>("hExtensionMixPurityNum", "Extension Mix Purity Numerator", effHistBins, xbins.data());
  mEExtensionMixFakeNum = createHistogram<TH1D>("hExtensionMixFakeNum", "Extension Mix Fake Numerator", effHistBins, xbins.data());
  // Patterns
  for (int i{0}; i < mBitPatternsBefore.size(); ++i) {
    mEExtensionPatternGoodNum[i] = createHistogram<TH1D>(fmt::format("hExtensionPatternGood_{:07b}", mBitPatternsBefore[i]).c_str(), fmt::format("Extended Tracks Pattern (GOOD) {:07b}", mBitPatternsBefore[i]).c_str(), effHistBins, xbins.data());
    mEExtensionPatternFakeNum[i] = createHistogram<TH1D>(fmt::format("hExtensionPatternFake_{:07b}", mBitPatternsBefore[i]).c_str(), fmt::format("Extended Tracks Pattern (FAKE) {:07b}", mBitPatternsBefore[i]).c_str(), effHistBins, xbins.data());
    for (int j{0}; j < mBitPatternsAfter.size(); ++j) {
      mEExtensionPatternIndGoodNum[i][j] = createHistogram<TH1D>(fmt::format("hExtensionPatternGood_{:07b}_{:07b}", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str(), fmt::format("Extended Tracks Pattern (GOOD) {:07b} -> {:07b}", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str(), effHistBins, xbins.data());
      mEExtensionPatternIndFakeNum[i][j] = createHistogram<TH1D>(fmt::format("hExtensionPatternFake_{:07b}_{:07b}", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str(), fmt::format("Extended Tracks Pattern (FAKE) {:07b} -> {:07b}", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str(), effHistBins, xbins.data());
    }
  }

  /// DCA
  mDCAxyVsPtPionsNormal = createHistogram<TH2D>("hDCAxyVsPtResNormal", "DCA_{#it{xy}} NORMAL Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", effHistBins, xbins.data(), 1000, -500, 500);
  mDCAxyVsPtPionsExtended = createHistogram<TH2D>("hDCAxyVsPtResExtended", "DCA_{#it{xy}} EXTENDED Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum)", effHistBins, xbins.data(), 1000, -500, 500);
  mDCAzVsPtPionsNormal = createHistogram<TH2D>("hDCAzVsPtResNormal", "DCA_{#it{z}} NORMAL Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", effHistBins, xbins.data(), 1000, -500, 500);
  mDCAzVsPtPionsExtended = createHistogram<TH2D>("hDCAzVsPtResExtended", "DCA_{#it{z}} EXTENDED Pions;#it{p}_{T} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum)", effHistBins, xbins.data(), 1000, -500, 500);

  mStream = std::make_unique<utils::TreeStreamRedirector>(mOutFileName.c_str(), "RECREATE");
}

void TrackExtensionStudy::run(ProcessingContext& pc)
{
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc);

  mTracksROFRecords = recoData.getITSTracksROFRecords();
  mTracks = recoData.getITSTracks();
  mTracksMCLabels = recoData.getITSTracksMCLabels();
  mClusters = recoData.getITSClusters();
  mClustersMCLCont = recoData.getITSClustersMCLabels();
  mInputITSidxs = recoData.getITSTracksClusterRefs();

  LOGP(info, "** Found in {} rofs:\n\t- {} clusters with {} labels\n\t- {} tracks with {} labels",
       mTracksROFRecords.size(), mClusters.size(), mClustersMCLCont->getIndexedSize(), mTracks.size(), mTracksMCLabels.size());
  LOGP(info, "** Found {} sources from kinematic files", mKineReader->getNSources());

  process();
}

void TrackExtensionStudy::process()
{
  LOGP(info, "** Filling particle table ... ");
  mParticleInfo.resize(mKineReader->getNSources()); // sources
  for (int iSource{0}; iSource < mKineReader->getNSources(); ++iSource) {
    mParticleInfo[iSource].resize(mKineReader->getNEvents(iSource)); // events
    for (int iEvent{0}; iEvent < mKineReader->getNEvents(iSource); ++iEvent) {
      const auto& mcEvent = mKineReader->getMCEventHeader(iSource, iEvent);
      mParticleInfo[iSource][iEvent].resize(mKineReader->getTracks(iSource, iEvent).size()); // tracks
      for (auto iPart{0}; iPart < mKineReader->getTracks(iEvent).size(); ++iPart) {
        const auto& part = mKineReader->getTracks(iSource, iEvent)[iPart];
        mParticleInfo[iSource][iEvent][iPart].eventX = mcEvent.GetX();
        mParticleInfo[iSource][iEvent][iPart].eventY = mcEvent.GetY();
        mParticleInfo[iSource][iEvent][iPart].eventZ = mcEvent.GetZ();
        mParticleInfo[iSource][iEvent][iPart].pdg = part.GetPdgCode();
        mParticleInfo[iSource][iEvent][iPart].pt = part.GetPt();
        mParticleInfo[iSource][iEvent][iPart].phi = part.GetPhi();
        mParticleInfo[iSource][iEvent][iPart].eta = part.GetEta();
        mParticleInfo[iSource][iEvent][iPart].vx = part.Vx();
        mParticleInfo[iSource][iEvent][iPart].vy = part.Vy();
        mParticleInfo[iSource][iEvent][iPart].vz = part.Vz();
        mParticleInfo[iSource][iEvent][iPart].isPrimary = part.isPrimary();
        mParticleInfo[iSource][iEvent][iPart].mother = part.getMotherTrackId();
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
        lab.get(trackID, evID, srcID, fake);
        auto& cluster = mClusters[iCluster];
        auto layer = mGeometry->getLayer(cluster.getSensorID());
        mParticleInfo[srcID][evID][trackID].clusters |= (1 << layer);
        if (fake) {
          mParticleInfo[srcID][evID][trackID].fakeClusters |= (1 << layer);
        }
      }
    }
  }

  LOGP(info, "** Analysing tracks ... ");
  int unaccounted{0}, good{0}, fakes{0}, extended{0};
  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
    const auto& lab = mTracksMCLabels[iTrack];
    if (!lab.isValid()) {
      unaccounted++;
      continue;
    }
    int trackID, evID, srcID;
    bool fake;
    lab.get(trackID, evID, srcID, fake);

    if (srcID == 99) { // skip QED
      unaccounted++;
      continue;
    }

    for (int iLayer{0}; iLayer < 7; ++iLayer) {
      if (mTracks[iTrack].isExtendedOnLayer(iLayer)) {
        ++extended;
        break;
      }
    }

    mParticleInfo[srcID][evID][trackID].isReco += !fake;
    mParticleInfo[srcID][evID][trackID].isFake += fake;
    if (mTracks[iTrack].isBetter(mParticleInfo[srcID][evID][trackID].track, 1.e9)) {
      mParticleInfo[srcID][evID][trackID].storedStatus = fake;
      mParticleInfo[srcID][evID][trackID].track = mTracks[iTrack];
      mParticleInfo[srcID][evID][trackID].mcTrack = *mKineReader->getTrack(lab);
    }
    fakes += fake;
    good += !fake;
  }
  LOGP(info, "** Some statistics:");
  LOGP(info, "\t- Total number of tracks: {}", mTracks.size());
  LOGP(info, "\t- Total number of tracks not corresponding to particles: {} ({:.2f} %)", unaccounted, unaccounted * 100. / mTracks.size());
  LOGP(info, "\t- Total number of fakes: {} ({:.2f} %)", fakes, fakes * 100. / mTracks.size());
  LOGP(info, "\t- Total number of good: {} ({:.2f} %)", good, good * 100. / mTracks.size());
  LOGP(info, "\t- Total number of extensions: {} ({:.2f} %)", extended, extended * 100. / mTracks.size());

  o2::dataformats::VertexBase collision;
  o2::dataformats::DCA impactParameter;
  LOGP(info, "** Filling histograms ... ");
  for (auto iTrack{0}; iTrack < mTracks.size(); ++iTrack) {
    auto& lab = mTracksMCLabels[iTrack];
    if (!lab.isValid()) {
      unaccounted++;
      continue;
    }
    int trackID, evID, srcID;
    bool fake;
    lab.get(trackID, evID, srcID, fake);
    const auto& part = mParticleInfo[srcID][evID][trackID];
    if (!part.isPrimary) {
      continue;
    }
    const auto& trk = part.track;
    bool isGood = part.isReco && !part.isFake;
    mHTrackCounts->Fill(0);

    std::bitset<7> extPattern{0};
    for (int iLayer{0}; iLayer < 7; ++iLayer) {
      if (trk.isExtendedOnLayer(iLayer)) {
        extPattern.set(iLayer);
      }
    }

    // Tree
    while (mWithTree) {
      constexpr float refRadius{70.f};
      constexpr float maxSnp{0.9f};
      auto cTrk = trk;
      if (!o2::base::Propagator::Instance()->PropagateToXBxByBz(cTrk, refRadius, maxSnp, 2.f, o2::base::Propagator::MatCorrType::USEMatCorrTGeo)) {
        break;
      }
      std::array<float, 3> xyz{(float)part.mcTrack.GetStartVertexCoordinatesX(), (float)part.mcTrack.GetStartVertexCoordinatesY(), (float)part.mcTrack.GetStartVertexCoordinatesZ()};
      std::array<float, 3> pxyz{(float)part.mcTrack.GetStartVertexMomentumX(), (float)part.mcTrack.GetStartVertexMomentumY(), (float)part.mcTrack.GetStartVertexMomentumZ()};
      auto pdg = O2DatabasePDG::Instance()->GetParticle(part.pdg);
      if (pdg == nullptr) {
        LOGP(error, "MC info not available");
        break;
      }
      auto mcTrk = o2::track::TrackPar(xyz, pxyz, TMath::Nint(pdg->Charge() / 3.), true);
      if (!mcTrk.rotate(cTrk.getAlpha()) || !o2::base::Propagator::Instance()->PropagateToXBxByBz(mcTrk, refRadius, maxSnp, 2.f, o2::base::Propagator::MatCorrType::USEMatCorrTGeo)) {
        break;
      }
      (*mStream) << "tree"
                 << "trk=" << cTrk
                 << "mcTrk=" << mcTrk
                 << "isGood=" << isGood
                 << "isExtended=" << extPattern.any()
                 << "\n";
      break;
    }

    // impact parameter
    while (isGood && std::abs(part.pdg) == 211) {
      auto trkC = part.track;
      collision.setXYZ(part.eventX, part.eventY, part.eventZ);
      if (!o2::base::Propagator::Instance()->propagateToDCA(collision, trkC, o2::base::Propagator::Instance()->getNominalBz(), 2.0, o2::base::Propagator::MatCorrType::USEMatCorrTGeo, &impactParameter)) {
        break;
      }

      auto dcaXY = impactParameter.getY() * 1e4;
      auto dcaZ = impactParameter.getZ() * 1e4;
      if (!extPattern.any()) {
        mDCAxyVsPtPionsNormal->Fill(part.pt, dcaXY);
        mDCAzVsPtPionsNormal->Fill(part.pt, dcaZ);
      } else {
        mDCAxyVsPtPionsExtended->Fill(part.pt, dcaXY);
        mDCAzVsPtPionsExtended->Fill(part.pt, dcaZ);
      }
      break;
    }

    mEExtensionDen->Fill(trk.getPt());

    if (!extPattern.any()) {
      mHTrackCounts->Fill(1);
      if (part.isReco || !part.isFake) {
        mHTrackCounts->Fill(2);
      } else {
        mHTrackCounts->Fill(3);
      }
      continue;
    }

    mHTrackCounts->Fill(4);
    mHLengthAny->Fill(trk.getNClusters());
    mHChi2Any->Fill(trk.getChi2());
    mHPtAny->Fill(trk.getPt());
    mEExtensionNum->Fill(trk.getPt());
    mEExtensionPurityDen->Fill(trk.getPt());
    mEExtensionFakeDen->Fill(trk.getPt());
    if (isGood) {
      mHTrackCounts->Fill(5);
      mHLengthGood->Fill(trk.getNClusters());
      mHChi2Good->Fill(trk.getChi2());
      mHPtGood->Fill(trk.getPt());
      mEExtensionPurityNum->Fill(trk.getPt());
    } else {
      mHTrackCounts->Fill(6);
      mHLengthFake->Fill(trk.getNClusters());
      mHChi2Fake->Fill(trk.getChi2());
      mHPtFake->Fill(trk.getPt());
      mEExtensionFakeNum->Fill(trk.getPt());
    }

    std::bitset<7> clusPattern{static_cast<uint8_t>(trk.getPattern())};
    for (int iLayer{0}; iLayer < 7; ++iLayer) {
      if (extPattern.test(iLayer)) {
        extPattern.set(iLayer);
        mHExtensionAny->Fill(iLayer);
        if (isGood) {
          mHExtensionGood->Fill(iLayer);
        } else {
          mHExtensionFake->Fill(iLayer);
        }
      }
    }
    std::bitset<7> oldPattern{clusPattern & ~extPattern}, holePattern{clusPattern};
    holePattern.flip();
    auto clusN = clusPattern.to_ulong();
    auto clusIdx = std::distance(std::begin(mBitPatternsAfter), std::find(std::begin(mBitPatternsAfter), std::end(mBitPatternsAfter), clusN));
    auto oldN = oldPattern.to_ulong();
    auto oldIdx = std::distance(std::begin(mBitPatternsBefore), std::find(std::begin(mBitPatternsBefore), std::end(mBitPatternsBefore), oldN));
    mHExtensionPatternsAny->Fill(oldIdx, clusIdx);
    if (isGood) {
      mHExtensionPatternsGood->Fill(oldIdx, clusIdx);
      mEExtensionPatternGoodNum[oldIdx]->Fill(trk.getPt());
      mEExtensionPatternIndGoodNum[oldIdx][clusIdx]->Fill(trk.getPt());
    } else {
      mHExtensionPatternsFake->Fill(oldIdx, clusIdx);
      mEExtensionPatternFakeNum[oldIdx]->Fill(trk.getPt());
      mEExtensionPatternIndFakeNum[oldIdx][clusIdx]->Fill(trk.getPt());
    }

    // old pattern
    bool oldFake{false}, newFake{false};
    for (int iLayer{0}; iLayer < 7; ++iLayer) {
      if (trk.isFakeOnLayer(iLayer)) {
        if (oldPattern.test(iLayer)) {
          oldFake = true;
        } else if (extPattern.test(iLayer)) {
          newFake = true;
        }
      }
    }
    if (oldFake && newFake) {
      mHTrackCounts->Fill(9);
      mEExtensionFakeMixNum->Fill(trk.getPt());
    } else if (oldFake) {
      mHTrackCounts->Fill(7);
      mEExtensionFakeBeforeNum->Fill(trk.getPt());
    } else if (newFake) {
      mHTrackCounts->Fill(8);
      mEExtensionFakeAfterNum->Fill(trk.getPt());
    }

    // Check if we missed some clusters
    if (isGood && holePattern.any()) {
      auto missPattern{clusPattern}, emptyPattern{clusPattern};
      for (int iLayer{0}; iLayer < 7; ++iLayer) {
        if (!holePattern.test(iLayer)) {
          continue;
        }

        // Check if there was actually a cluster that we missed
        if ((part.clusters & (1 << iLayer)) != 0) {
          missPattern.set(iLayer);
        } else {
          emptyPattern.set(iLayer);
        }
      }

      if (missPattern != clusPattern) {
        auto missN = missPattern.to_ulong();
        auto missIdx = std::distance(std::begin(mBitPatternsAfter), std::find(std::begin(mBitPatternsAfter), std::end(mBitPatternsAfter), missN));
        mHExtensionPatternsGoodMissed->Fill(clusIdx, missIdx);
      }
      if (emptyPattern != clusPattern) {
        auto emptyN = emptyPattern.to_ulong();
        auto emptyIdx = std::distance(std::begin(mBitPatternsAfter), std::find(std::begin(mBitPatternsAfter), std::end(mBitPatternsAfter), emptyN));
        mHExtensionPatternsGoodEmpty->Fill(clusIdx, emptyIdx);
      }
    }

    // Top/Bot/Mixed Extension
    bool isTop = (extPattern & mTopMask).any();
    bool isBot = (extPattern & mBotMask).any();
    if (isTop && isBot) {
      mEExtensionMixNum->Fill(trk.getPt());
      if (isGood) {
        mEExtensionMixPurityNum->Fill(trk.getPt());
      } else {
        mEExtensionMixFakeNum->Fill(trk.getPt());
      }
    } else if (isTop) {
      mEExtensionTopNum->Fill(trk.getPt());
      if (isGood) {
        mEExtensionTopPurityNum->Fill(trk.getPt());
      } else {
        mEExtensionBotFakeNum->Fill(trk.getPt());
      }
    } else {
      mEExtensionBotNum->Fill(trk.getPt());
      if (isGood) {
        mEExtensionBotPurityNum->Fill(trk.getPt());
      } else {
        mEExtensionMixFakeNum->Fill(trk.getPt());
      }
    }
  }
}

void TrackExtensionStudy::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    mGeometry = GeometryTGeo::Instance();
    mGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

void TrackExtensionStudy::endOfStream(EndOfStreamContext& ec)
{
  LOGP(info, "Writing results to {}", mOutFileName);
  mStream->GetFile()->cd();
  for (const auto h : mHistograms) {
    h->Write();
  }

  LOGP(info, "Calculating efficiencies");
  auto makeEff = [](auto num, auto den, const char* name, const char* title) {
    auto e = std::make_unique<TEfficiency>(*num, *den);
    e->SetName(name);
    e->SetTitle(title);
    e->Write();
  };
  makeEff(mEExtensionNum.get(), mEExtensionDen.get(), "eExtension", "Track Extension EXT TRK/ALL");
  makeEff(mEExtensionPurityNum.get(), mEExtensionPurityDen.get(), "eExtensionPurity", "Track Extension Purity GOOD/EXT TRK");
  makeEff(mEExtensionFakeNum.get(), mEExtensionFakeDen.get(), "eExtensionFake", "Track Extension Fake FAKE/EXT TRK");
  makeEff(mEExtensionFakeBeforeNum.get(), mEExtensionFakeNum.get(), "eExtensionFakeBefore", "Track Extension Fake FAKE BEF/FAKE EXT TRK");
  makeEff(mEExtensionFakeAfterNum.get(), mEExtensionFakeNum.get(), "eExtensionFakeAfter", "Track Extension Fake FAKE AFT/FAKE EXT TRK");
  makeEff(mEExtensionFakeMixNum.get(), mEExtensionFakeNum.get(), "eExtensionFakeMix", "Track Extension Fake FAKE MIX/FAKE EXT TRK");
  makeEff(mEExtensionTopNum.get(), mEExtensionDen.get(), "eExtensionTop", "Track Extension Top");
  makeEff(mEExtensionTopPurityNum.get(), mEExtensionPurityDen.get(), "eExtensionTopPurity", "Track Extension Purity GOOD TOP/EXT TRK");
  makeEff(mEExtensionTopFakeNum.get(), mEExtensionFakeNum.get(), "eExtensionTopFake", "Track Extension FAKE TOP/EXT FAKE TRK");
  makeEff(mEExtensionBotNum.get(), mEExtensionDen.get(), "eExtensionBot", "Track Extension Bot");
  makeEff(mEExtensionBotPurityNum.get(), mEExtensionPurityDen.get(), "eExtensionBotPurity", "Track Extension Purity GOOD BOT/EXT TRK");
  makeEff(mEExtensionBotFakeNum.get(), mEExtensionFakeNum.get(), "eExtensionBotFake", "Track Extension FAKE BOT/EXT FAKE TRK");
  makeEff(mEExtensionMixNum.get(), mEExtensionDen.get(), "eExtensionMix", "Track Extension Mix");
  makeEff(mEExtensionMixPurityNum.get(), mEExtensionPurityDen.get(), "eExtensionMixPurity", "Track Extension Purity GOOD MIX/EXT TRK");
  makeEff(mEExtensionMixFakeNum.get(), mEExtensionFakeNum.get(), "eExtensionMixFake", "Track Extension FAKE MIX/EXT FAKE TRK");
  for (int i{0}; i < mBitPatternsBefore.size(); ++i) {
    makeEff(mEExtensionPatternGoodNum[i].get(), mEExtensionPurityNum.get(), fmt::format("eExtensionPatternGood_{:07b}", mBitPatternsBefore[i]).c_str(), fmt::format("Extended Tracks Pattern (GOOD) {:07b} GOOD EXT TRK/EXT TRK", mBitPatternsBefore[i]).c_str());
    makeEff(mEExtensionPatternFakeNum[i].get(), mEExtensionFakeNum.get(), fmt::format("eExtensionPatternFake_{:07b}", mBitPatternsBefore[i]).c_str(), fmt::format("Extended Tracks Pattern (FAKE) {:07b} FAKE EXT TRK/EXT TRK", mBitPatternsBefore[i]).c_str());
    for (int j{0}; j < mBitPatternsAfter.size(); ++j) {
      makeEff(mEExtensionPatternIndGoodNum[i][j].get(), mEExtensionPatternGoodNum[i].get(), fmt::format("eExtensionPatternGood_{:07b}_{:07b}", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str(), fmt::format("Extended Tracks Pattern (GOOD) {:07b} -> {:07b} GOOD EXT TRK/EXT TRK", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str());
      makeEff(mEExtensionPatternIndFakeNum[i][j].get(), mEExtensionPatternFakeNum[i].get(), fmt::format("eExtensionPatternFake_{:07b}_{:07b}", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str(), fmt::format("Extended Tracks Pattern (FAKE) {:07b} -> {:07b} FAKE EXT TRK/EXT TRK", mBitPatternsBefore[i], mBitPatternsAfter[j]).c_str());
    }
  }

  mStream->Close();
}

void TrackExtensionStudy::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
}

DataProcessorSpec getTrackExtensionStudy(mask_t srcTracksMask, mask_t srcClustersMask, std::shared_ptr<o2::steer::MCKinematicsReader> kineReader)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(srcTracksMask, true);
  dataRequest->requestClusters(srcClustersMask, true);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  return DataProcessorSpec{
    "its-study-track-extension",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackExtensionStudy>(dataRequest, srcTracksMask, kineReader, ggRequest)},
    Options{{"with-tree", o2::framework::VariantType::Bool, false, {"Produce in addition a tree"}}}};
}

} // namespace o2::its::study
