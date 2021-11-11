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
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
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
#include "DataFormatsTOF/Cluster.h"
#include "TOFReconstruction/EventTimeMaker.h"
//#include "GlobalTracking/MatchTOF.h"
#include "GlobalTrackingWorkflow/TOFEventTimeChecker.h"

#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TString.h"
#include "TProfile.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"

using namespace o2::framework;
// using MCLabelsTr = gsl::span<const o2::MCCompLabel>;
// using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

using evIdx = o2::dataformats::EvIndex<int, int>;
using MatchOutputType = std::vector<o2::dataformats::MatchInfoTOF>;
using GID = o2::dataformats::GlobalTrackID;

struct MyTrack : o2::tof::eventTimeTrackTest {
  double tofSignalDouble() const { return mSignalDouble; }
  double mSignalDouble = 0.0;
};

using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;

bool MyFilter(const MyTrack& tr)
{
  return (tr.mP < 2.0);
} // accept all

namespace o2
{
namespace globaltracking
{

class TOFEventTimeChecker : public Task
{
 public:
  TOFEventTimeChecker(std::shared_ptr<DataRequest> dr, bool useMC) : mDataRequest(dr), mUseMC(useMC) {}
  ~TOFEventTimeChecker() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void fillMatching(GID gid);
  void processEvent(std::vector<MyTrack>& tracks);

 private:
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
  RecoContainer mRecoData;
  std::shared_ptr<DataRequest> mDataRequest;
  bool mUseMC = true;
  TStopwatch mTimer;
};

void TOFEventTimeChecker::processEvent(std::vector<MyTrack>& tracks)
{
  int nBC = int(tracks[0].tofSignalDouble() * o2::tof::Geo::BC_TIME_INPS_INV);
  for (auto& track : tracks) {
    track.mSignal = float(track.mSignalDouble - double(o2::tof::Geo::BC_TIME_INPS) * nBC);
  }

  auto evtime = o2::tof::evTimeMaker<std::vector<MyTrack>, MyTrack, MyFilter>(tracks);
  //
  const float cinv = 33.35641;

  int nt = 0;
  for (auto& track : tracks) {
    float et = evtime.eventTime;
    float erret = evtime.eventTimeError;
    float res = sqrt(track.tofExpSigmaPi() * track.tofExpSigmaPi() + erret * erret);

    float sumw = 1. / erret / erret;
    et *= sumw;
    et -= evtime.weights[nt] * evtime.tracktime[nt];
    sumw -= evtime.weights[nt];
    et /= sumw;
    erret = sqrt(1. / sumw);

    nt++;

    //Beta
    float beta = track.mLength / (track.tofSignal() - et) * cinv;

    float betaexpPi = track.mLength / (track.tofExpTimePi()) * cinv;
    float betaexpKa = track.mLength / (track.tofExpTimeKa()) * cinv;
    float betaexpPr = track.mLength / (track.tofExpTimePr()) * cinv;

    //Mass
    float mass = track.mP / beta * TMath::Sqrt(TMath::Abs(1 - beta * beta));
    float massexpPi = track.mP / betaexpPi * TMath::Sqrt(TMath::Abs(1 - betaexpPi * betaexpPi));
    float massexpKa = track.mP / betaexpKa * TMath::Sqrt(TMath::Abs(1 - betaexpKa * betaexpKa));
    float massexpPr = track.mP / betaexpPr * TMath::Sqrt(TMath::Abs(1 - betaexpPr * betaexpPr));

    //Fill histos
    mHTimePi->Fill(track.tofSignal() - et - track.tofExpTimePi());
    mHTimeKa->Fill(track.tofSignal() - et - track.tofExpTimeKa());
    mHTimePr->Fill(track.tofSignal() - et - track.tofExpTimePr());
    mHMass->Fill(mass);
    mHMassExpPi->Fill(massexpPi);
    mHMassExpKa->Fill(massexpKa);
    mHMassExpPr->Fill(massexpPr);
    mHBetavsP->Fill(track.mP, beta);
    mPBetavsPExpPi->Fill(track.mP, betaexpPi);
    mPBetavsPExpKa->Fill(track.mP, betaexpKa);
    mPBetavsPExpPr->Fill(track.mP, betaexpPr);
    mHTimePivsP->Fill(track.mP, track.tofSignal() - et - track.tofExpTimePi());
    mHTimeKvsP->Fill(track.mP, track.tofSignal() - et - track.tofExpTimeKa());
    mHTimePrvsP->Fill(track.mP, track.tofSignal() - et - track.tofExpTimePr());
    mHMassvsP->Fill(track.mP, mass);
    mPMassvsPExpPi->Fill(track.mP, massexpPi);
    mPMassvsPExpKa->Fill(track.mP, massexpKa);
    mPMassvsPExpPr->Fill(track.mP, massexpPr);
    if (track.mP > 0.7 && track.mP < 1.1) {
      mHTimevsResEvtimePi->Fill(erret, track.tofSignal() - et - track.tofExpTimePi());
    }
    mHEventTimevsResEvtime->Fill(erret, et);
  }
}

void TOFEventTimeChecker::fillMatching(GID gid)
{
  if (!gid.includesDet(DetID::TOF)) {
    return;
  }
  const o2::dataformats::MatchInfoTOF& match = mRecoData.getTOFMatch(gid);
  const o2::track::TrackLTIntegral& info = match.getLTIntegralOut();

  if (info.getL() < 370) {
    return;
  }

  MyTrack trk;
  int trksource = 5;
  if (gid.getSource() == GID::TPCTOF) {
    const auto& array = mRecoData.getTPCTracks();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getP();
    trksource = 0;
  } else if (gid.getSource() == GID::ITSTPCTOF) {
    const auto& array = mRecoData.getTPCITSTracks();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getPt();
    trksource = 1;
  } else if (gid.getSource() == GID::TPCTRDTOF) {
    const auto& array = mRecoData.getTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getPt();
    trksource = 2;
  } else if (gid.getSource() == GID::ITSTPCTRDTOF) {
    const auto& array = mRecoData.getITSTPCTRDTracks<o2::trd::TrackTRD>();
    GID gTrackId = match.getTrackRef();
    const auto& srctrk = array[gTrackId.getIndex()];
    trk.mPt = srctrk.getPt();
    trk.mP = srctrk.getPt();
    trksource = 3;
  }

  const char* sources[5] = {"TPC", "ITS-TPC", "TPC-TRD", "ITS-TPC-TRD", "NONE"};

  trk.expTimes[0] = info.getTOF(2); // pi
  trk.expTimes[1] = info.getTOF(3); // ka
  trk.expTimes[2] = info.getTOF(4); // pr
  trk.expSigma[0] = 120;            // dummy resolution (to be updated)
  trk.expSigma[1] = 120;            // dummy resolution (to be updated)
  trk.expSigma[2] = 120;            // dummy resolution (to be updated)

  int tofcl = match.getIdxTOFCl();
  //  trk.mSignal = mTOFClustersArrayInp[tofcl].getTime();
  double tofsignal = match.getSignal();

  trk.mSignalDouble = tofsignal;

  //trk.mSignal = match.getSignal();
  trk.mTOFChi2 = match.getChi2();
  trk.mLength = info.getL();
  //  trk.mHypo = 0;

  if (mSlewing) { // let's calibrate
    int ch = mTOFClustersArrayInp[tofcl].getMainContributingChannel();

    if (mSlewing->isProblematic(ch)) {
      LOG(DEBUG) << "skip channel " << ch << " since problematic";
      return;
    }
    float tot = mTOFClustersArrayInp[tofcl].getTot();
    trk.mSignalDouble -= mSlewing->evalTimeSlewing(ch, tot);
    LOG(DEBUG) << "calibration -> " << mSlewing->evalTimeSlewing(ch, tot);
  }

  mMyTracks.push_back(trk);
}

void TOFEventTimeChecker::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry("", false);
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};

  TFile* fsleewing = TFile::Open("localTimeSlewing.root");
  if (fsleewing) {
    mSlewing = (TimeSlewing*)fsleewing->Get("ccdb_object");
  }

  mHTimePi = new TH1F("HTimePi", ";t_{TOF} - t_{exp}^{#pi} (ps)", 2500, -25000, 25000);
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
  mHEventTimevsResEvtime = new TH2F("HEventTimevsResEvtime", ";TOF event time resolution (ps); TOF event time (ps)", 100, 0, 200, 1000, -2000, 2000);
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

  LOG(DEBUG) << "isTrackSourceLoaded: TPC -> " << mIsTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTOF) << ")";
  LOG(DEBUG) << "isTrackSourceLoaded: ITSTPC -> " << mIsITSTPC << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTOF) << ")";
  LOG(DEBUG) << "isTrackSourceLoaded: TPCTRD -> " << mIsTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::TPCTRDTOF) << ")";
  LOG(DEBUG) << "isTrackSourceLoaded: ITSTPCTRD -> " << mIsITSTPCTRD << " (t=" << mRecoData.isTrackSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ",m=" << mRecoData.isMatchSourceLoaded(o2::dataformats::GlobalTrackID::Source::ITSTPCTRDTOF) << ")";
  LOG(DEBUG) << "TOF cluster size = " << mTOFClustersArrayInp.size();

  if (!mTOFClustersArrayInp.size()) {
    return;
  }

  auto creator = [this](auto& trk, GID gid, float time0, float terr) {
    this->fillMatching(gid);
    return true;
  };
  mRecoData.createTracksVariadic(creator);

  // sorting matching in time
  std::sort(mMyTracks.begin(), mMyTracks.end(),
            [](MyTrack a, MyTrack b) { return a.tofSignalDouble() < b.tofSignalDouble(); });

  for (auto& element : mMyTracks) { // loop print
    LOG(DEBUG) << "Time cluster = " << element.tofSignal() << " ps - pt = " << element.pt();
  }

  std::vector<MyTrack> tracks;
  for (int i = 0; i < mMyTracks.size(); i++) { // loop looking for interaction candidates
    tracks.clear();
    int ntrk = 1;
    double time = mMyTracks[i].tofSignalDouble();
    tracks.emplace_back(mMyTracks[i]);
    for (; i < mMyTracks.size(); i++) {
      double timeCurrent = mMyTracks[i].tofSignalDouble();
      if (timeCurrent - time > 25E3) {
        i--;
        break;
      }
      tracks.emplace_back(mMyTracks[i]);
      ntrk++;
    }
    if (ntrk > 2) { // good candidate with time
      processEvent(tracks);
    }
  }

  mTimer.Stop();
}

void TOFEventTimeChecker::endOfStream(EndOfStreamContext& ec)
{
  LOGF(DEBUG, "TOF matching total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);

  TFile* write = new TFile("TOFperformance.root", "recreate");
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
}

DataProcessorSpec getTOFEventTimeCheckerSpec(GID::mask_t src, bool useMC)
{
  auto dataRequest = std::make_shared<DataRequest>();

  // request TOF clusters
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(GID::getSourceMask(GID::TOF), useMC);
  dataRequest->requestTOFMatches(src, useMC);

  return DataProcessorSpec{
    "tof-eventime",
    dataRequest->inputs,
    {},
    AlgorithmSpec{adaptFromTask<TOFEventTimeChecker>(dataRequest, useMC)},
    Options{}};
}

} // namespace globaltracking
} // namespace o2
