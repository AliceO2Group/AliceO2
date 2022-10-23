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

/// \file RecoContainerCreateTracksVariadic.h
/// \brief Wrapper container for different reconstructed object types
/// \author ruben.shahoyan@cern.chM

#include "Framework/ProcessingContext.h"
#include "Framework/InputSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMFT/TrackMFT.h"

#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/ROFRecord.h"

#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/Track.h"
#include "DataFormatsMID/MCClusterLabel.h"
#include "DataFormatsMID/MCLabel.h"

#include "ReconstructionDataFormats/TrackMCHMID.h"

#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsHMP/Cluster.h"
#include "DataFormatsHMP/Trigger.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFV0/RecPoints.h"
#include "DataFormatsFDD/RecPoint.h"
#include "DataFormatsZDC/RecEvent.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "DataFormatsCTP/Digits.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/MatchInfoHMP.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "DataFormatsCPV/Cluster.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "DataFormatsEMCAL/MCLabel.h"

#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

//________________________________________________________
template <class T>
void o2::globaltracking::RecoContainer::createTracksVariadic(T creator, GTrackID::mask_t srcSel) const
{
  // We go from most complete tracks to least complete ones, taking into account that some track times
  // do not bear their own kinematics but just constrain the time
  // As we get more track types functional, this method should be completed
  // If user provided function creator returns true, then the track is considered as consumed and its contributing
  // simpler tracks will not be provided to the creator. If it returns false, the creator will be called also
  // with this simpler contrubutors.
  // The creator function is called with track kinematics, track GlobalTrackID and track timing information as 2 floats
  // which depends on the track time:
  // 1) For track types containing TimeStampWithError ts it is ts.getTimeStamp(), getTimeStampError()
  // 2) For tracks with asymmetric time uncertainty, e.g. TPC: as mean time of t0-errBwd,t+errFwd and 0.5(errBwd+errFwd), all in TPC time bins
  // 3) For tracks whose timing is provided as RO frame: as time in \mus for RO frame start since the start of TF, half-duration of RO window.

  auto start_time = std::chrono::high_resolution_clock::now();
  constexpr float PS2MUS = 1e-6;
  std::array<std::vector<uint8_t>, GTrackID::NSources> usedData;
  auto flagUsed2 = [&usedData](int idx, int src) {
    if (!usedData[src].empty()) {
      usedData[src][idx] = 1;
    }
  };
  auto flagUsed = [&usedData, &flagUsed2](const GTrackID gidx) { flagUsed2(gidx.getIndex(), gidx.getSource()); };
  auto isUsed2 = [&usedData](int idx, int src) { return (!usedData[src].empty()) && (usedData[src][idx] != 0); };
  auto isUsed = [&usedData, isUsed2](const GTrackID gidx) { return isUsed2(gidx.getIndex(), gidx.getSource()); };

  // create only for those data types which are used
  const auto tracksITS = getITSTracks();
  const auto trkITABRefs = getITSABRefs();
  const auto tracksMFT = getMFTTracks();
  const auto tracksMCH = getMCHTracks();
  const auto tracksMID = getMIDTracks();
  const auto tracksTPC = getTPCTracks();
  const auto tracksTPCITS = getTPCITSTracks();
  const auto tracksMFTMCH = getGlobalFwdTracks();
  const auto matchesMCHMID = getMCHMIDMatches();
  const auto tracksTPCTOF = getTPCTOFTracks();   // TOF-TPC tracks with refit
  const auto matchesTPCTOF = getTPCTOFMatches(); // and corresponding matches
  const auto tracksTPCTRD = getTPCTRDTracks<o2::trd::TrackTRD>();
  const auto matchesITSTPCTOF = getITSTPCTOFMatches();       // just matches, no refit done
  const auto matchesTPCTRDTOF = getTPCTRDTOFMatches();       // just matches, no refit done
  const auto matchesITSTPCTRDTOF = getITSTPCTRDTOFMatches(); // just matches, no refit done
  const auto tofClusters = getTOFClusters();
  const auto tracksITSTPCTRD = getITSTPCTRDTracks<o2::trd::TrackTRD>();

  const auto trigTPCTRD = getTPCTRDTriggers();

  usedData[GTrackID::ITS].resize(tracksITS.size());                         // to flag used ITS tracks
  usedData[GTrackID::MCH].resize(tracksMCH.size());                         // to flag used MCH tracks
  usedData[GTrackID::MFT].resize(tracksMFT.size());                         // to flag used MFT tracks
  usedData[GTrackID::MID].resize(tracksMID.size());                         // to flag used MFT tracks
  usedData[GTrackID::TPC].resize(tracksTPC.size());                         // to flag used TPC tracks
  usedData[GTrackID::ITSTPC].resize(tracksTPCITS.size());                   // to flag used ITSTPC tracks
  usedData[GTrackID::MFTMCH].resize(tracksMFTMCH.size());                   // to flag used MFTMCH tracks
  usedData[GTrackID::ITSTPCTRD].resize(tracksITSTPCTRD.size());             // to flag used ITSTPCTRD tracks
  usedData[GTrackID::TPCTRD].resize(tracksTPCTRD.size());                   // to flag used TPCTRD tracks
  usedData[GTrackID::ITSTPCTRD].resize(tracksITSTPCTRD.size());             // to flag used TPCTRD tracks
  usedData[GTrackID::ITSTPCTOF].resize(getITSTPCTOFMatches().size());       // to flag used ITSTPC-TOF matches
  usedData[GTrackID::TPCTRDTOF].resize(getTPCTRDTOFMatches().size());       // to flag used ITSTPC-TOF matches
  usedData[GTrackID::ITSTPCTRDTOF].resize(getITSTPCTRDTOFMatches().size()); // to flag used ITSTPC-TOF matches

  static int BCDiffErrCount = 0;
  constexpr int MAXBCDiffErrCount = 5;
  GTrackID::Source currentSource = GTrackID::NSources;
  auto getBCDiff = [startIR = this->startIR, &currentSource](const o2::InteractionRecord& ir) {
    auto bcd = ir.differenceInBC(startIR);
    if (uint64_t(bcd) > o2::constants::lhc::LHCMaxBunches * 256 && BCDiffErrCount < MAXBCDiffErrCount) {
      LOGP(alarm, "ATTENTION: wrong bunches diff. {} for current IR {} wrt 1st TF orbit {}, source:{}", bcd, ir.asString(), startIR.asString(), GTrackID::getSourceName(currentSource));
      BCDiffErrCount++;
    }
    return bcd;
  };

  //  ITS-TPC-TRD-TOF
  {
    currentSource = GTrackID::ITSTPCTRDTOF;
    if (srcSel[currentSource]) {
      if (matchesITSTPCTRDTOF.size() && (!tofClusters.size() || !tracksITSTPCTRD.size())) {
        throw std::runtime_error(fmt::format("Global-TOF tracks ({}) require ITS-TPC-TRD tracks ({}) and TOF clusters ({})",
                                             matchesITSTPCTRDTOF.size(), tracksITSTPCTRD.size(), tofClusters.size()));
      }
      for (unsigned i = 0; i < matchesITSTPCTRDTOF.size(); i++) {
        const auto& match = matchesITSTPCTRDTOF[i];
        auto gidx = match.getTrackRef(); // this should be corresponding ITS-TPC-TRD track
        // no need to check isUsed: by construction this ITS-TPC-TRD was not used elsewhere
        const auto& tofCl = tofClusters[match.getTOFClIndex()];
        float timeTOFMUS = (tofCl.getTime() - match.getLTIntegralOut().getTOF(o2::track::PID::Pion)) * PS2MUS; // tof time in \mus, FIXME: account for time of flight to R TOF
        const float timeErr = 0.010f;                                                                          // assume 10 ns error FIXME
        if (creator(tracksITSTPCTRD[gidx.getIndex()], {i, currentSource}, timeTOFMUS, timeErr)) {
          flagUsed(gidx); // flag used ITS-TPC-TRD tracks
        }
      }
    }
  }

  // TPC-TRD-TOF
  {
    currentSource = GTrackID::TPCTRDTOF;
    if (srcSel[currentSource]) {
      if (matchesTPCTRDTOF.size() && (!tofClusters.size() || !tracksTPCTRD.size())) {
        throw std::runtime_error(fmt::format("Global-TOF tracks ({}) require TPC-TRD tracks ({}) and TOF clusters ({})",
                                             matchesTPCTRDTOF.size(), tracksTPCTRD.size(), tofClusters.size()));
      }
      for (unsigned i = 0; i < matchesTPCTRDTOF.size(); i++) {
        const auto& match = matchesTPCTRDTOF[i];
        auto gidx = match.getTrackRef(); // this should be corresponding TPC-TRD track
        const auto& tofCl = tofClusters[match.getTOFClIndex()];
        float timeTOFMUS = (tofCl.getTime() - match.getLTIntegralOut().getTOF(o2::track::PID::Pion)) * PS2MUS; // tof time in \mus, FIXME: account for time of flight to R TOF
        const float timeErr = 0.010f;                                                                          // assume 10 ns error FIXME
        if (creator(tracksTPCTRD[gidx.getIndex()], {i, currentSource}, timeTOFMUS, timeErr)) {
          flagUsed(gidx); // flag used TPC-TRD tracks
        }
      }
    }
  }

  // ITS-TPC-TRD
  {
    currentSource = GTrackID::ITSTPCTRD;
    if (srcSel[currentSource]) {
      const auto trigITSTPCTRD = getITSTPCTRDTriggers();
      for (unsigned itr = 0; itr < trigITSTPCTRD.size(); itr++) {
        const auto& trig = trigITSTPCTRD[itr];
        auto bcdiff = getBCDiff(trig.getBCData());
        float t0 = bcdiff * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
        for (unsigned int i = trig.getTrackRefs().getFirstEntry(); i < (unsigned int)trig.getTrackRefs().getEntriesBound(); i++) {
          const auto& trc = tracksITSTPCTRD[i];
          if (isUsed2(i, currentSource)) {
            flagUsed(trc.getRefGlobalTrackId()); // flag seeding ITS-TPC track
            continue;
          }
          if (creator(trc, {i, currentSource}, t0, 1e-3)) { // assign 1ns error to BC
            flagUsed(trc.getRefGlobalTrackId());            // flag seeding ITS-TPC track
          }
        }
      }
    }
  }

  // ITS-TPC-TOF matches, these are just MatchInfoTOF objects, pointing on ITS-TPC match and TOF cl.
  {
    currentSource = GTrackID::ITSTPCTOF;
    if (srcSel[currentSource]) {
      if (matchesITSTPCTOF.size() && (!tofClusters.size() || !tracksTPCITS.size())) {
        throw std::runtime_error(fmt::format("Global-TOF tracks ({}) require ITS-TPC tracks ({}) and TOF clusters ({})",
                                             matchesITSTPCTOF.size(), tracksTPCITS.size(), tofClusters.size()));
      }
      for (unsigned i = 0; i < matchesITSTPCTOF.size(); i++) {
        const auto& match = matchesITSTPCTOF[i];
        auto gidx = match.getTrackRef(); // this should be corresponding ITS-TPC track
        if (isUsed(gidx)) {              // RS FIXME: THIS IS TEMPORARY, until the TOF matching will use ITS-TPC-TRD as an input
          continue;
        }
        // no need to check isUsed: by construction this ITS-TPC was not used elsewhere
        const auto& tofCl = tofClusters[match.getTOFClIndex()];
        float timeTOFMUS = (tofCl.getTime() - match.getLTIntegralOut().getTOF(o2::track::PID::Pion)) * PS2MUS; // tof time in \mus, FIXME: account for time of flight to R TOF
        const float timeErr = 0.010f;                                                                          // assume 10 ns error FIXME
        if (creator(tracksTPCITS[gidx.getIndex()], {i, currentSource}, timeTOFMUS, timeErr)) {
          flagUsed(gidx); // flag used ITS-TPC tracks
        }
      }
    }
  }

  // TPC-TRD
  {
    currentSource = GTrackID::TPCTRD;
    if (srcSel[currentSource]) {
      const auto trigTPCTRD = getTPCTRDTriggers();
      for (unsigned itr = 0; itr < trigTPCTRD.size(); itr++) {
        const auto& trig = trigTPCTRD[itr];
        auto bcdiff = getBCDiff(trig.getBCData());
        float t0 = bcdiff * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
        for (unsigned int i = trig.getTrackRefs().getFirstEntry(); i < (unsigned int)trig.getTrackRefs().getEntriesBound(); i++) {
          const auto& trc = tracksTPCTRD[i];
          if (isUsed2(i, currentSource)) {
            flagUsed(trc.getRefGlobalTrackId()); // flag seeding TPC track
            continue;
          }
          if (creator(trc, {i, currentSource}, t0, 1e-3)) { // assign 1ns error to BC
            flagUsed(trc.getRefGlobalTrackId());            // flag seeding TPC track
          }
        }
      }
    }
  }

  // ITS-TPC matches
  {
    currentSource = GTrackID::ITSTPC;
    if (srcSel[currentSource]) {
      for (unsigned i = 0; i < tracksTPCITS.size(); i++) {
        const auto& matchTr = tracksTPCITS[i];
        if (isUsed2(i, currentSource)) {
          flagUsed(matchTr.getRefITS()); // flag used ITS tracks or AB tracklets (though the latter is not really necessary)
          flagUsed(matchTr.getRefTPC()); // flag used TPC tracks
          continue;
        }
        if (creator(matchTr, {i, currentSource}, matchTr.getTimeMUS().getTimeStamp(), matchTr.getTimeMUS().getTimeStampError())) {
          flagUsed(matchTr.getRefITS()); // flag used ITS tracks or AB tracklets (though the latter is not really necessary)
          flagUsed(matchTr.getRefTPC()); // flag used TPC tracks
        }
      }
    }
  }

  // TPC-TOF matches
  {
    currentSource = GTrackID::TPCTOF;
    if (srcSel[currentSource]) {
      if (matchesTPCTOF.size() && !tracksTPCTOF.size()) {
        throw std::runtime_error(fmt::format("TPC-TOF matched tracks ({}) require TPCTOF matches ({}) and TPCTOF tracks ({})",
                                             matchesTPCTOF.size(), tracksTPCTOF.size()));
      }
      for (unsigned i = 0; i < matchesTPCTOF.size(); i++) {
        const auto& match = matchesTPCTOF[i];
        const auto& gidx = match.getTrackRef(); // TPC track global idx
        if (isUsed(gidx)) {                     // flag used TPC tracks
          continue;
        }
        const auto& trc = tracksTPCTOF[i];
        if (creator(trc, {i, currentSource}, trc.getTimeMUS().getTimeStamp(), trc.getTimeMUS().getTimeStampError())) {
          flagUsed(gidx); // flag used TPC tracks
        }
      }
    }
  }

  // MFT-MCH tracks
  {
    currentSource = GTrackID::MFTMCH;
    if (srcSel[currentSource]) {
      for (unsigned i = 0; i < tracksMFTMCH.size(); i++) {
        const auto& matchTr = tracksMFTMCH[i];
        if (creator(matchTr, {i, currentSource}, matchTr.getTimeMUS().getTimeStamp(), matchTr.getTimeMUS().getTimeStampError())) {
          flagUsed2(matchTr.getMFTTrackID(), GTrackID::MFT);
          flagUsed2(matchTr.getMCHTrackID(), GTrackID::MCH);
        }
      }
    }
  }

  // MCH-MID matches
  {
    currentSource = GTrackID::MCHMID;
    if (srcSel[currentSource]) {
      if (matchesMCHMID.size() && !tracksMCH.size()) {
        throw std::runtime_error(fmt::format("MCH-MID matched tracks ({}) require MCHMID matches ({}) and MCH tracks ({})",
                                             matchesMCHMID.size(), tracksMCH.size()));
      }
      for (unsigned i = 0; i < matchesMCHMID.size(); i++) {
        const auto& match = matchesMCHMID[i];
        auto [trcTime, isInTF] = match.getTimeMUS(startIR, 256, BCDiffErrCount < MAXBCDiffErrCount);
        if (!isInTF && BCDiffErrCount < MAXBCDiffErrCount) {
          BCDiffErrCount++;
        }
        auto gidxMCH = match.getMCHRef();
        const auto& trc = tracksMCH[gidxMCH.getIndex()];
        if (creator(trc, {i, currentSource}, trcTime.getTimeStamp(), trcTime.getTimeStampError())) {
          flagUsed(gidxMCH);           // flag used MCH tracks
          flagUsed(match.getMIDRef()); // flag used MID tracks (if requested)
        }
      }
    }
  }

  // ITS only tracks
  {
    currentSource = GTrackID::ITS;
    if (srcSel[currentSource]) {
      const auto& rofrs = getITSTracksROFRecords();
      for (unsigned irof = 0; irof < rofrs.size(); irof++) {
        const auto& rofRec = rofrs[irof];
        auto bcdiff = getBCDiff(rofRec.getBCData());
        float t0 = bcdiff * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
        int trlim = rofRec.getFirstEntry() + rofRec.getNEntries();
        for (int it = rofRec.getFirstEntry(); it < trlim; it++) {
          if (isUsed2(it, currentSource)) { // skip used tracks
            continue;
          }
          GTrackID gidITS(it, currentSource);
          const auto& trc = tracksITS[it];
          creator(trc, gidITS, t0, 0.5);
        }
      }
    }
  }

  // MFT only tracks
  {
    currentSource = GTrackID::MFT;
    if (srcSel[currentSource]) {
      const auto& rofrs = getMFTTracksROFRecords();
      for (unsigned irof = 0; irof < rofrs.size(); irof++) {
        const auto& rofRec = rofrs[irof];
        auto bcdiff = getBCDiff(rofRec.getBCData());
        float t0 = bcdiff * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
        int trlim = rofRec.getFirstEntry() + rofRec.getNEntries();
        for (int it = rofRec.getFirstEntry(); it < trlim; it++) {
          if (isUsed2(it, currentSource)) {
            continue;
          }
          GTrackID gidMFT(it, currentSource);
          const auto& trc = tracksMFT[it];
          creator(trc, gidMFT, t0, 0.5);
        }
      }
    }
  }

  // MCH standalone tracks
  {
    currentSource = GTrackID::MCH;
    if (srcSel[currentSource]) {
      const auto& rofs = getMCHTracksROFRecords();
      for (const auto& rof : rofs) {
        if (rof.getNEntries() == 0) {
          continue;
        }
        auto [trcTime, isInTF] = rof.getTimeMUS(startIR, 256, BCDiffErrCount < MAXBCDiffErrCount);
        if (!isInTF && BCDiffErrCount < MAXBCDiffErrCount) {
          BCDiffErrCount++;
        }
        for (int idx = rof.getFirstIdx(); idx <= rof.getLastIdx(); ++idx) {
          if (isUsed2(idx, currentSource)) {
            continue;
          }
          GTrackID gidMCH(idx, currentSource);
          const auto& trc = tracksMCH[idx];
          creator(trc, gidMCH, trcTime.getTimeStamp(), trcTime.getTimeStampError());
        }
      }
    }
  }

  // MID standalone tracks
  {
    currentSource = GTrackID::MID;
    if (srcSel[currentSource]) {
      const auto& rofs = getMIDTracksROFRecords();
      for (const auto& rof : rofs) {
        if (rof.nEntries == 0) {
          continue;
        }
        auto [trcTime, isInTF] = rof.getTimeMUS(startIR, 256, BCDiffErrCount < MAXBCDiffErrCount);
        if (!isInTF && BCDiffErrCount < MAXBCDiffErrCount) {
          BCDiffErrCount++;
        }
        if (trcTime.getTimeStamp() < 0.f) {
          if (BCDiffErrCount - 1 < MAXBCDiffErrCount) {
            LOGP(alarm, "Skipping MID ROF with {} entries since it precedes TF start", rof.nEntries);
          }
          continue;
        }
        for (int idx = rof.firstEntry; idx < rof.getEndIndex(); ++idx) {
          if (isUsed2(idx, currentSource)) {
            continue;
          }
          GTrackID gidMID(idx, currentSource);
          const auto& trc = tracksMID[idx];
          creator(trc, gidMID, trcTime.getTimeStamp(), trcTime.getTimeStampError());
        }
      }
    }
  }

  // TPC only tracks
  {
    currentSource = GTrackID::TPC;
    if (srcSel[currentSource]) {
      int nacc = 0, noffer = 0;
      for (unsigned i = 0; i < tracksTPC.size(); i++) {
        if (isUsed2(i, currentSource)) { // skip used tracks
          continue;
        }
        const auto& trc = tracksTPC[i];
        creator(trc, {i, currentSource}, trc.getTime0() + 0.5 * (trc.getDeltaTFwd() - trc.getDeltaTBwd()), 0.5 * (trc.getDeltaTFwd() + trc.getDeltaTBwd()));
      }
    }
  }

  auto current_time = std::chrono::high_resolution_clock::now();
  LOG(info) << "RecoContainer::createTracks took " << std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time).count() * 1e-6 << " CPU s.";
}

template <class T>
inline constexpr auto isITSTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::its::TrackITS>;
}

template <class T>
inline constexpr auto isITSABRef()
{
  return std::is_same_v<std::decay_t<T>, o2::itsmft::TrkClusRef>;
}

template <class T>
inline constexpr auto isMFTTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::mft::TrackMFT>;
}

template <class T>
inline constexpr auto isMCHTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::mch::TrackMCH>;
}

template <class T>
inline constexpr auto isMIDTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::mid::Track>;
}

template <class T>
inline constexpr auto isTPCTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::tpc::TrackTPC>;
}

template <class T>
inline constexpr auto isTRDTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::trd::TrackTRD>;
}

template <class T>
inline constexpr auto isTPCTOFTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::dataformats::TrackTPCTOF>;
}

template <class T>
inline constexpr auto isTPCITSTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::dataformats::TrackTPCITS>;
}

template <class T>
inline constexpr auto isGlobalFwdTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::dataformats::GlobalFwdTrack>;
}

template <class T>
inline constexpr auto isBarrelTrack()
{
  return std::is_base_of<o2::track::TrackParF, std::decay_t<T>>::value || std::is_base_of<o2::track::TrackParD, std::decay_t<T>>::value;
}
