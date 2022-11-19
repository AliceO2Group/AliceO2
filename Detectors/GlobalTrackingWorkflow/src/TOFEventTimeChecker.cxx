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

/// @file   TOFEventTimeChecker.cxx

#include <vector>
#include <string>
#include "TStopwatch.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/NameConf.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"

// from Tracks
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/GlobalTrackAccessor.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/TrackTRD.h"

// from TOF
#include "TOFBase/Geo.h"
#include "TOFBase/Utils.h"
#include "DataFormatsTOF/Cluster.h"
#include "TOFBase/EventTimeMaker.h"
//#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/TOFEventTimeChecker.h"
#include "DetectorsBase/GRPGeomHelper.h"

#include "TSystem.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TString.h"
#include "TProfile.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"

#define TDEBUG

using namespace o2::framework;
// using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
// using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;
using GID = o2::dataformats::GlobalTrackID;

struct MyTrack : o2::tof::eventTimeTrackTest {
  double tofSignalDouble() const { return mSignalDouble; }
  float tofExpSignalDe() const { return mExpDe; }
  double mSignalDouble = 0.0;
  float mEta = 0.0;
  float mPTPC = 0.0;
  float mPhi = 0.0;
  float mExpDe = 0;
  int mIsProb = 0;
  int mCh = -1;
  float mChi2 = 0;
  bool mHasTOF = false;
  int mSource = -1;
  double mTrktime = 0;
  double mTrktimeRes = 0;
  float mDx = 0;
  float mDz = 0;
};

using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;

bool MyFilter(const MyTrack& tr)
{
  return (tr.mP < 2.0 && tr.mEta > o2::tof::Utils::mEtaMin && tr.mEta < o2::tof::Utils::mEtaMax && tr.mHasTOF && tr.mSource >= 0);
} // accept all

namespace o2
{
namespace globaltracking
{

class TOFEventTimeChecker : public Task
{
 public:
  TOFEventTimeChecker(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC) : mDataRequest(dr), mGGCCDBRequest(gr), mUseMC(useMC) {}
  ~TOFEventTimeChecker() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void fillMatching(GID gid, float time0, float time0res);
  void processEvent(std::vector<MyTrack>& tracks);
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      return;
    }
  }

 private:
  void updateTimeDependentParams(ProcessingContext& pc)
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    static bool initOnceDone = false;
    if (!initOnceDone) { // this params need to be queried only once
      initOnceDone = true;
      const auto bcs = o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getBunchFilling().getFilledBCs();
      for (auto bc : bcs) {
        o2::tof::Utils::addInteractionBC(bc, true);
      }
    }
  }

  bool mIsTPC;
  bool mIsTPCTRD;
  bool mIsITSTPCTRD;
  bool mIsITSTPC;
  gsl::span<const o2::tof::Cluster> mTOFClustersArrayInp; ///< input TOF clusters
  std::vector<MyTrack> mMyTracks;
  TimeSlewing* mSlewing = nullptr;
  TH1F* mHTimePi;
  TH1F* mHTimeKa;
  TH1F* mHTimePr;
  TH1F* mHMass;
  TH1F* mHMassExpPi;
  TH1F* mHMassExpKa;
  TH1F* mHMassExpPr;
  TH2F* mHBetavsP;
  TH2F* mHTimePivsP;
  TH2F* mHTimeKvsP;
  TH2F* mHTimePrvsP;
  TProfile* mPBetavsPExpPi;
  TProfile* mPBetavsPExpKa;
  TProfile* mPBetavsPExpPr;
  TH2F* mHMassvsP;
  TProfile* mPMassvsPExpPi;
  TProfile* mPMassvsPExpKa;
  TProfile* mPMassvsPExpPr;
  TH2F* mHTimevsResEvtimePi;
  TH2F* mHEventTimevsResEvtime;
  TFile* mFout;
#ifdef TDEBUG
  TTree* mTree;
#endif
  int mOrbit = 0;
  int mCh;
  float mP = 0;
  float mPt = 0;
  float mPTPC = 0;
  float mEta = 0;
  float mPhi = 0;
  float mChi2 = 0;
  float mL = 0;
  float mTof = 0;
  float mT0 = 0;
  float mT0Res = 0;
  float mExpDe = 0;
  float mExpPi = 0;
  float mExpKa = 0;
  float mExpPr = 0;
  float mDx = 0;
  float mDz = 0;
  int mIsProb = 0;
  int mSource = -1;
  float mTrktime = 0;
  float mTrktimeRes = 0;
  RecoContainer mRecoData;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC = true;
  TStopwatch mTimer;
};

void TOFEventTimeChecker::processEvent(std::vector<MyTrack>& tracks)
{
  int nBC = int(tracks[0].tofSignalDouble() * o2::tof::Geo::BC_TIME_INPS_INV);
  for (auto& track : tracks) {
    if (!o2::tof::Utils::hasFillScheme()) {
      track.mSignal = float(track.mSignalDouble - double(o2::tof::Geo::BC_TIME_INPS) * nBC);
      track.mTrktime = (track.mTrktime - double(o2::tof::Geo::BC_TIME_INPS) * nBC);
    } else {
      double localTime = track.mSignalDouble;

      // get into orbit
      int bcStarOrbit = int(localTime * o2::tof::Geo::BC_TIME_INPS_INV);
      bcStarOrbit = (bcStarOrbit / o2::constants::lhc::LHCMaxBunches) * o2::constants::lhc::LHCMaxBunches; // truncation
      localTime -= bcStarOrbit * o2::tof::Geo::BC_TIME_INPS;
      int mask = 0;
      track.mSignal = o2::tof::Utils::subtractInteractionBC(localTime, mask);
      localTime = track.mTrktime;
      bcStarOrbit = int(localTime * o2::tof::Geo::BC_TIME_INPS_INV);
      bcStarOrbit = (bcStarOrbit / o2::constants::lhc::LHCMaxBunches) * o2::constants::lhc::LHCMaxBunches; // truncation
      localTime -= bcStarOrbit * o2::tof::Geo::BC_TIME_INPS;
      mask = 0;
      track.mTrktime = o2::tof::Utils::subtractInteractionBC(localTime, mask);
    }
  }

  auto evtime = o2::tof::evTimeMaker<std::vector<MyTrack>, MyTrack, MyFilter>(tracks);

  if (evtime.mEventTime - o2::tof::Utils::mLHCPhase < -2000 || evtime.mEventTime - o2::tof::Utils::mLHCPhase > 2000) {
    return;
  }
  //
  const float cinv = 33.35641;

  int nt = 0;
  for (auto& track : tracks) {
    mT0 = evtime.mEventTime;
    mT0Res = evtime.mEventTimeError;

    float sumw = 1. / (mT0Res * mT0Res);
    mT0 *= sumw;
    mT0 -= evtime.mWeights[nt] * evtime.mTrackTimes[nt];
    sumw -= evtime.mWeights[nt];
    mT0 /= sumw;
    mT0Res = sqrt(1. / sumw);

    nt++;

    mCh = track.mCh;
    mP = track.mP;
    mPt = track.mPt;
    mPTPC = track.mPTPC;
    mEta = track.mEta;
    mSource = track.mSource;
    mPhi = track.mPhi;
    mChi2 = track.mChi2;
    mL = track.mLength;
    mTof = track.tofSignal();
    mExpDe = track.tofExpSignalDe();
    mExpPi = track.tofExpSignalPi();
    mExpKa = track.tofExpSignalKa();
    mExpPr = track.tofExpSignalPr();
    mIsProb = track.mIsProb;
    mTrktime = track.mTrktime;
    mTrktimeRes = track.mTrktimeRes;
    mDx = track.mDx;
    mDz = track.mDz;

#ifdef TDEBUG
    mTree->Fill();
#endif

    // remove unphysical tracks
    if (mTof - mT0 - mExpPi < -5000) {
      continue;
    }

    //Beta
    float beta = mL / (mTof - mT0) * cinv;

    float betaexpPi = mL / mExpPi * cinv;
    float betaexpKa = mL / mExpKa * cinv;
    float betaexpPr = mL / mExpPr * cinv;

    //Mass
    float mass = mP / beta * TMath::Sqrt(TMath::Abs(1 - beta * beta));
    float massexpPi = mP / betaexpPi * TMath::Sqrt(TMath::Abs(1 - betaexpPi * betaexpPi));
    float massexpKa = mP / betaexpKa * TMath::Sqrt(TMath::Abs(1 - betaexpKa * betaexpKa));
    float massexpPr = mP / betaexpPr * TMath::Sqrt(TMath::Abs(1 - betaexpPr * betaexpPr));

    if (massexpPi < 0.13) { // remove wrong track lengths
      continue;
    }

    //Fill histos
    mHTimePi->Fill(mTof - mT0 - mExpPi);
    mHTimeKa->Fill(mTof - mT0 - mExpKa);
    mHTimePr->Fill(mTof - mT0 - mExpPr);
    mHMass->Fill(mass);
    mHMassExpPi->Fill(massexpPi);
    mHMassExpKa->Fill(massexpKa);
    mHMassExpPr->Fill(massexpPr);
    mHBetavsP->Fill(mP, beta);
    mPBetavsPExpPi->Fill(mP, betaexpPi);
    mPBetavsPExpKa->Fill(mP, betaexpKa);
    mPBetavsPExpPr->Fill(mP, betaexpPr);
    mHTimePivsP->Fill(mP, mTof - mT0 - mExpPi);
    mHTimeKvsP->Fill(mP, mTof - mT0 - mExpKa);
    mHTimePrvsP->Fill(mP, mTof - mT0 - mExpPr);
    mHMassvsP->Fill(mP, mass);
    mPMassvsPExpPi->Fill(mP, massexpPi);
    mPMassvsPExpKa->Fill(mP, massexpKa);
    mPMassvsPExpPr->Fill(mP, massexpPr);
    if (mP > 0.7 && mP < 1.1) {
      mHTimevsResEvtimePi->Fill(mT0Res, mTof - mT0 - mExpPi);
    }
    mHEventTimevsResEvtime->Fill(mT0Res, mT0);
  }
}

void TOFEventTimeChecker::fillMatching(GID gid, float time0, float time0res)
{
  MyTrack trk;
  trk.mHasTOF = true;

  if (!gid.includesDet(DetID::TOF)) {
    trk.mHasTOF = false;
  }

  trk.mTrktime = time0 * 1E6;
  trk.mTrktimeRes = time0res * 1E6;

  trk.mDx = 0;
  trk.mDz = 0;

  int trksource = 5;
  if (gid.getSource() == GID::TPCTOF) {
    const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
    const auto& array = mRecoData.getTPCTracks();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trk.mDx = match.getDXatTOF();
    trk.mDz = match.getDZatTOF();
    trksource = 0;
  } else if (gid.getSource() == GID::TPC) {
    const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
    const auto& array = mRecoData.getTPCTracks();
    GID gTrackId = gid;
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trksource = 0;
  } else if (gid.getSource() == GID::ITSTPCTOF) {
    const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
    const auto& array = mRecoData.getTPCITSTracks();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trksource = 1;
    trk.mTrktime = srctrk.getTimeMUS().getTimeStamp() * 1E6;
    trk.mTrktimeRes = srctrk.getTimeMUS().getTimeStampError() * 1E6;
    trk.mDx = match.getDXatTOF();
    trk.mDz = match.getDZatTOF();
  } else if (gid.getSource() == GID::ITSTPC) {
    const auto& array = mRecoData.getTPCITSTracks();
    GID gTrackId = gid; //match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trksource = 1;
    trk.mTrktime = srctrk.getTimeMUS().getTimeStamp() * 1E6;
    trk.mTrktimeRes = srctrk.getTimeMUS().getTimeStampError() * 1E6;
  } else if (gid.getSource() == GID::TPCTRDTOF) {
    const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
    const auto& array = mRecoData.getTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trksource = 2;
    trk.mDx = match.getDXatTOF();
    trk.mDz = match.getDZatTOF();
  } else if (gid.getSource() == GID::TPCTRD && 0) {
    const auto& array = mRecoData.getTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = gid; // match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trksource = 2;
  } else if (gid.getSource() == GID::ITSTPCTRDTOF) {
    const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
    const auto& array = mRecoData.getITSTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trksource = 3;
    trk.mDx = match.getDXatTOF();
    trk.mDz = match.getDZatTOF();
  } else if (gid.getSource() == GID::ITSTPCTRD && 0) {
    const auto& array = mRecoData.getTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = gid; // match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt() * srctrk.getCharge();
    trk.mPTPC = srctrk.getP();
    trk.mP = srctrk.getP();
    trk.mEta = srctrk.getEta();
    trk.mPhi = srctrk.getPhi();
    trksource = 3;
  }

  trk.mSource = trksource;

  const char* sources[5] = {"TPC", "ITS-TPC", "TPC-TRD", "ITS-TPC-TRD", "NONE"};

  if (trk.mHasTOF) {
    const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
    const o2::track::TrackLTIntegral& info = match.getLTIntegralOut();

    if (info.getL() < 370) {
      trk.mHasTOF = false;
    }

    trk.mExpDe = info.getTOF(5);      // el
    trk.expTimes[0] = info.getTOF(2); // pi
    trk.expTimes[1] = info.getTOF(3); // ka
    trk.expTimes[2] = info.getTOF(4); // pr
    trk.expSigma[0] = 120;            // dummy resolution (to be updated)
    trk.expSigma[1] = 120;            // dummy resolution (to be updated)
    trk.expSigma[2] = 120;            // dummy resolution (to be updated)

    trk.mChi2 = match.getChi2();

    int tofcl = match.getIdxTOFCl();
    //  trk.mSignal = mTOFClustersArrayInp[tofcl].getTime();
    double tofsignal = match.getSignal();

    trk.mSignalDouble = tofsignal;

    //trk.mSignal = match.getSignal();
    trk.mTOFChi2 = match.getChi2();
    trk.mLength = info.getL();
    //  trk.mHypo = 0;
    trk.mCh = mTOFClustersArrayInp[tofcl].getMainContributingChannel();

    if (mSlewing) { // let's calibrate
      trk.mIsProb = mSlewing->isProblematic(trk.mCh);
      if (mSlewing->isProblematic(trk.mCh)) {
        //      LOG(debug) << "skip channel " << trk.mCh << " since problematic";
        //      return;
      }
      float tot = mTOFClustersArrayInp[tofcl].getTot();
      trk.mSignalDouble -= mSlewing->evalTimeSlewing(trk.mCh, tot);
      LOG(debug) << "calibration -> " << mSlewing->evalTimeSlewing(trk.mCh, tot);
    }
  }
  if (!trk.mHasTOF) {
    trk.mExpDe = 0;
    trk.expTimes[0] = 0;
    trk.expTimes[1] = 0;
    trk.expTimes[2] = 0;
    trk.expSigma[0] = 0; // dummy resolution (to be updated)
    trk.expSigma[1] = 0; // dummy resolution (to be updated)
    trk.expSigma[2] = 0; // dummy resolution (to be updated)
    trk.mSignalDouble = time0 * 1E6;
    trk.mTOFChi2 = 9999;
    trk.mLength = 0;
    trk.mCh = -1;
    trk.mChi2 = 0;
    trk.mIsProb = 0;
  }

  mMyTracks.push_back(trk);
}

void TOFEventTimeChecker::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  // extrct orbit from dir
  std::string dir = gSystem->GetWorkingDirectory();
  if (dir.find("orbit") < dir.size()) {
    dir.erase(0, dir.find("orbit") + 5);
    dir.erase(dir.find("_"), dir.size());
    while (dir.size() && dir[0] == '0') {
      dir.erase(0, 1);
    }
    sscanf(dir.c_str(), "%d", &mOrbit);
  }

  TFile* fsleewing = TFile::Open("localTimeSlewing.root");
  if (fsleewing) {
    mSlewing = (TimeSlewing*)fsleewing->Get("ccdb_object");
  }

  mFout = new TFile("TOFperformance.root", "recreate");

  mHTimePi = new TH1F("HTimePi", ";t_{TOF} - t_{exp}^{#pi} (ps)", 500, -5000, 5000);
  mHTimeKa = new TH1F("HTimeKa", ";t_{TOF} - t_{exp}^{K} (ps)", 500, -5000, 5000);
  mHTimePr = new TH1F("HTimePr", ";t_{TOF} - t_{exp}^{p} (ps)", 500, -5000, 5000);
  mHMass = new TH1F("HMass", ";M (GeV/#it{c}^{2})", 1000, 0, 2.);
  mHMassExpPi = new TH1F("HMassExpPi", ";M(#beta_{exp}^{#pi}) (GeV/#it{c}^{2})", 1000, 0, 2.);
  mHMassExpKa = new TH1F("HMassExpKa", ";M(#beta_{exp}^{K}) (GeV/#it{c}^{2})", 1000, 0, 2.);
  mHMassExpPr = new TH1F("HMassExpPr", ";M(#beta_{exp}^{p}) (GeV/#it{c}^{2})", 1000, 0, 2.);
  mHBetavsP = new TH2F("HBetavsP", ";#it{p} (GeV/#it{c});TOF #beta", 1000, 0., 5, 1000, 0., 1.5);
  mPBetavsPExpPi = new TProfile("PBetavsPExpPi", ";#it{p} (GeV/#it{c}); #beta_{exp}^{#pi}", 1000, 0., 5, 0., 1.5);
  mPBetavsPExpKa = new TProfile("PBetavsPExpKa", ";#it{p} (GeV/#it{c}); #beta_{exp}^{K}", 1000, 0., 5, 0., 1.5);
  mPBetavsPExpPr = new TProfile("PBetavsPExpPr", ";#it{p} (GeV/#it{c}); #beta_{exp}^{p}", 1000, 0., 5, 0., 1.5);
  mHTimePivsP = new TH2F("HTimePivsP", ";#it{p} (GeV/#it{c});t_{TOF} - t_{exp}^{#pi} (ps)", 500, 0., 5, 500, -5000, 5000);
  mHTimeKvsP = new TH2F("HTimeKavsP", ";#it{p} (GeV/#it{c});t_{TOF} - t_{exp}^{K} (ps)", 500, 0., 5, 500, -5000, 5000);
  mHTimePrvsP = new TH2F("HTimePrvsP", ";#it{p} (GeV/#it{c});t_{TOF} - t_{exp}^{p} (ps)", 500, 0., 5, 500, -5000, 5000);
  mHMassvsP = new TH2F("HMassvsP", ";#it{p} (GeV/#it{c}); M (GeV/#it{c}^{2})", 1000, 0., 5, 1000, 0., 2.);
  mPMassvsPExpPi = new TProfile("PMassvsPExpPi", ";#it{p} (GeV/#it{c}); M(#beta_{exp}^{#pi}) [GeV/#it{c}^{2}]", 1000, 0., 5, 0., 2.);
  mPMassvsPExpKa = new TProfile("PMassvsPExpKa", ";#it{p} (GeV/#it{c}); M(#beta_{exp}^{K}) [GeV/#it{c}^{2}]", 1000, 0., 5, 0., 2.);
  mPMassvsPExpPr = new TProfile("PMassvsPExpPr", ";#it{p} (GeV/#it{c}); M(#beta_{exp}^{p}) [GeV/#it{c}^{2}]", 1000, 0., 5, 0., 2.);
  mHTimevsResEvtimePi = new TH2F("HTimevsResEvtimePi", "0.7 < p < 1.1 GeV/#it{c};TOF event time resolution (ps);t_{TOF} - t_{exp}^{#pi} (ps)", 200, 0., 200, 500, -5000, 5000);
  mHEventTimevsResEvtime = new TH2F("HEventTimevsResEvtime", ";TOF event time resolution (ps); TOF event time (ps)", 100, 0, 200, 5000, -20000, 20000);

#ifdef TDEBUG
  mTree = new TTree("tree", "tree");
  mTree->Branch("orbit", &mOrbit, "orbit/I");
  mTree->Branch("ch", &mCh, "ch/I");
  mTree->Branch("isProb", &mIsProb, "isProb/I");
  mTree->Branch("p", &mP, "p/F");
  mTree->Branch("pt", &mPt, "pt/F");
  mTree->Branch("pTPC", &mPTPC, "pTPC/F");
  mTree->Branch("source", &mSource, "source/I");
  mTree->Branch("eta", &mEta, "eta/F");
  mTree->Branch("phi", &mPhi, "phi/F");
  mTree->Branch("chi2", &mChi2, "chi2/F");
  mTree->Branch("l", &mL, "l/F");
  mTree->Branch("tof", &mTof, "tof/F");
  mTree->Branch("t0", &mT0, "t0/F");
  mTree->Branch("t0res", &mT0Res, "t0res/F");
  mTree->Branch("trkTime", &mTrktime, "trkTime/F");
  mTree->Branch("trkTimeRes", &mTrktimeRes, "trkTimeRes/F");
  mTree->Branch("dx", &mDx, "dx/F");
  mTree->Branch("dz", &mDz, "dz/F");
  mTree->Branch("expDe", &mExpDe, "expDe/F");
  mTree->Branch("expPi", &mExpPi, "expPi/F");
  mTree->Branch("expKa", &mExpKa, "expKa/F");
  mTree->Branch("expPr", &mExpPr, "expPr/F");
#endif
}

void TOFEventTimeChecker::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  mMyTracks.clear();
  mRecoData.collectData(pc, *mDataRequest.get());

  mIsTPC = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF));
  mIsITSTPC = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF));
  mIsITSTPCTRD = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRD) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF));
  mIsTPCTRD = (mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRD) && mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF));

  mTOFClustersArrayInp = mRecoData.getTOFClusters();

  LOG(debug) << "isTrackSourceLoaded: TPC -> " << mIsTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ")";
  LOG(debug) << "isTrackSourceLoaded: ITSTPC -> " << mIsITSTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ")";
  LOG(debug) << "isTrackSourceLoaded: TPCTRD -> " << mIsTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ")";
  LOG(debug) << "isTrackSourceLoaded: ITSTPCTRD -> " << mIsITSTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ")";
  LOG(debug) << "TOF cluster size = " << mTOFClustersArrayInp.size();

  if (!mTOFClustersArrayInp.size()) {
    return;
  }
  updateTimeDependentParams(pc);

  auto creator = [this](auto& trk, GID gid, float time0, float terr) {
    this->fillMatching(gid, time0, terr);
    return true;
  };
  mRecoData.createTracksVariadic(creator);

  // sorting matching in time
  std::sort(mMyTracks.begin(), mMyTracks.end(),
            [](MyTrack a, MyTrack b) { return a.tofSignalDouble() < b.tofSignalDouble(); });

  for (auto& element : mMyTracks) { // loop print
    LOG(debug) << "Time cluster = " << element.tofSignal() << " ps - pt = " << element.pt();
  }

  std::vector<MyTrack> tracks;
  for (int i = 0; i < mMyTracks.size(); i++) { // loop looking for interaction candidates
    tracks.clear();
    int ntrk = 1;
    double time = mMyTracks[i].tofSignalDouble();
    tracks.emplace_back(mMyTracks[i]);
    for (; i < mMyTracks.size(); i++) {
      double timeCurrent = mMyTracks[i].tofSignalDouble();
      if (timeCurrent - time > 100E3) {
        i--;
        break;
      }
      tracks.emplace_back(mMyTracks[i]);
      ntrk++;
    }
    if (ntrk > 0) { // good candidate with time
      processEvent(tracks);
    }
  }

  mTimer.Stop();
}

void TOFEventTimeChecker::endOfStream(EndOfStreamContext& ec)
{
  LOGF(debug, "TOF matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);

  mFout->cd();
#ifdef TDEBUG
  mTree->Write();
#endif
  mHTimePi->Write();
  mHTimeKa->Write();
  mHTimePr->Write();
  mHMass->Write();
  mHMassExpPi->Write();
  mHMassExpKa->Write();
  mHMassExpPr->Write();
  mHBetavsP->Write();
  mPBetavsPExpPi->Write();
  mPBetavsPExpKa->Write();
  mPBetavsPExpPr->Write();
  mHTimePivsP->Write();
  mHTimeKvsP->Write();
  mHTimePrvsP->Write();
  mHMassvsP->Write();
  mPMassvsPExpPi->Write();
  mPMassvsPExpKa->Write();
  mPMassvsPExpPr->Write();
  mHTimevsResEvtimePi->Write();
  mHEventTimevsResEvtime->Write();
  mFout->Close();
}

DataProcessorSpec getTOFEventTimeCheckerSpec(GID::mask_t src, bool useMC)
{
  auto dataRequest = std::make_shared<DataRequest>();

  // request TOF clusters
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(GID::getSourceMask(GID::TOF), useMC);
  dataRequest->requestTOFMatches(src, useMC);
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              false,                          // GRPECS=true
                                                              true,                           // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "tof-eventime",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<TOFEventTimeChecker>(dataRequest, ggRequest, useMC)},
    Options{}};
}

} // namespace globaltracking
} // namespace o2
