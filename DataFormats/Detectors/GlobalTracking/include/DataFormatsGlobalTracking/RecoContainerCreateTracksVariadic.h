// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoContainerCreateTracksVariadic.h
/// \brief Wrapper container for different reconstructed object types
/// \author ruben.shahoyan@cern.ch

#include "Framework/ProcessingContext.h"
#include "Framework/InputSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTOF/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

//________________________________________________________
template <class T>
void o2::globaltracking::RecoContainer::createTracksVariadic(T creator) const
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
  const auto tracksMFT = getMFTTracks();
  const auto tracksTPC = getTPCTracks();
  const auto tracksTPCITS = getTPCITSTracks();
  const auto tracksTPCTOF = getTPCTOFTracks();   // TOF-TPC tracks with refit
  const auto matchesTPCTOF = getTPCTOFMatches(); // and corresponding matches
  const auto tracksTPCTRD = getTPCTRDTracks<o2::trd::TrackTRD>();
  const auto matchesITSTPCTOF = getTOFMatches(); // just matches, no refit done
  const auto tofClusters = getTOFClusters();
  const auto tracksITSTPCTRD = getITSTPCTRDTracks<o2::trd::TrackTRD>();

  const auto trigTPCTRD = getTPCTRDTriggers();

  usedData[GTrackID::ITS].resize(tracksITS.size());                                      // to flag used ITS tracks
  usedData[GTrackID::MFT].resize(tracksMFT.size());                                      // to flag used MFT tracks
  usedData[GTrackID::TPC].resize(tracksTPC.size());                                      // to flag used TPC tracks
  usedData[GTrackID::ITSTPC].resize(tracksTPCITS.size());                                // to flag used ITSTPC tracks
  usedData[GTrackID::ITSTPCTRD].resize(tracksITSTPCTRD.size());                          // to flag used ITSTPCTRD tracks
  usedData[GTrackID::TPCTRD].resize(tracksTPCTRD.size());                                // to flag used TPCTRD tracks
  usedData[GTrackID::TOF].resize(getTOFMatches().size());                                // to flag used ITSTPC-TOF matches

  // ITS-TPC-TRD-TOF
  // TODO, will flag used ITS-TPC-TRD

  // ITS-TPC-TRD
  {
    const auto trigITSTPCTRD = getITSTPCTRDTriggers();
    for (unsigned itr = 0; itr < trigITSTPCTRD.size(); itr++) {
      const auto& trig = trigITSTPCTRD[itr];
      float t0 = trig.getBCData().differenceInBC(startIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
      for (unsigned i = trig.getTrackRefs().getFirstEntry(); i < trig.getTrackRefs().getEntriesBound(); i++) {
        const auto& trc = tracksITSTPCTRD[i];
        if (isUsed2(i, GTrackID::ITSTPCTRD)) {
          flagUsed(trc.getRefGlobalTrackId()); // flag seeding ITS-TPC track
          continue;
        }
        if (creator(trc, {i, GTrackID::ITSTPCTRD}, t0, 1e-3)) { // assign 1ns error to BC
          flagUsed2(i, GTrackID::ITSTPCTRD);                    // flag itself (is it needed?)
          flagUsed(trc.getRefGlobalTrackId());                  // flag seeding ITS-TPC track
        }
      }
    }
  }

  // ITS-TPC-TOF matches, thes are just MatchInfoTOF objects, pointing on ITS-TPC match and TOF cl.
  {

    if (matchesITSTPCTOF.size() && (!tofClusters.size() || !tracksTPCITS.size())) {
      throw std::runtime_error(fmt::format("Global-TOF tracks ({}) require ITS-TPC tracks ({}) and TOF clusters ({})",
                                           matchesITSTPCTOF.size(), tracksTPCITS.size(), tofClusters.size()));
    }
    for (unsigned i = 0; i < matchesITSTPCTOF.size(); i++) {
      const auto& match = matchesITSTPCTOF[i];
      auto gidx = match.getEvIdxTrack().getIndex(); // this should be corresponding ITS-TPC track
      if (isUsed(gidx)) {                           // RS FIXME: THIS IS TEMPORARY, until the TOF matching will use ITS-TPC-TRD as an input
        continue;
      }
      // no need to check isUsed: by construction this ITS-TPC was not used elsewhere
      const auto& tofCl = tofClusters[match.getTOFClIndex()];
      float timeTOFMUS = (tofCl.getTime() - match.getLTIntegralOut().getTOF(o2::track::PID::Pion)) * PS2MUS; // tof time in \mus, FIXME: account for time of flight to R TOF
      const float timeErr = 0.010f;                                                                          // assume 10 ns error FIXME
      if (creator(tracksTPCITS[gidx.getIndex()], {i, GTrackID::ITSTPCTOF}, timeTOFMUS, timeErr)) {
        flagUsed2(i, GTrackID::TOF); // flag used TOF match // TODO might be not needed
        flagUsed(gidx);              // flag used ITS-TPC tracks
      }
    }
  }

  // TPC-TRD
  {
    const auto trigTPCTRD = getTPCTRDTriggers();
    for (unsigned itr = 0; itr < trigTPCTRD.size(); itr++) {
      const auto& trig = trigTPCTRD[itr];
      float t0 = trig.getBCData().differenceInBC(startIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
      for (unsigned i = trig.getTrackRefs().getFirstEntry(); i < trig.getTrackRefs().getEntriesBound(); i++) {
        const auto& trc = tracksTPCTRD[i];
        if (isUsed2(i, GTrackID::TPCTRD)) {
          flagUsed(trc.getRefGlobalTrackId()); // flag seeding TPC track
          continue;
        }
        if (creator(trc, {i, GTrackID::TPCTRD}, t0, 1e-6)) { // assign 1ns error to BC
          flagUsed2(i, GTrackID::TPCTRD);                    // flag itself (is it needed?)
          flagUsed(trc.getRefGlobalTrackId());               // flag seeding TPC track
        }
      }
    }
  }

  // ITS-TPC matches, may refer to ITS, TPC (TODO: something else?) tracks
  {
    for (unsigned i = 0; i < tracksTPCITS.size(); i++) {
      const auto& matchTr = tracksTPCITS[i];
      if (isUsed2(i, GTrackID::ITSTPC)) {
        flagUsed(matchTr.getRefITS()); // flag used ITS tracks
        flagUsed(matchTr.getRefTPC()); // flag used TPC tracks
        continue;
      }
      if (creator(matchTr, {i, GTrackID::ITSTPC}, matchTr.getTimeMUS().getTimeStamp(), matchTr.getTimeMUS().getTimeStampError())) {
        flagUsed2(i, GTrackID::ITSTPC);
        flagUsed(matchTr.getRefITS()); // flag used ITS tracks
        flagUsed(matchTr.getRefTPC()); // flag used TPC tracks
      }
    }
  }

  // TPC-TOF matches, may refer to TPC (TODO: something else?) tracks
  {
    if (matchesTPCTOF.size() && !tracksTPCTOF.size()) {
      throw std::runtime_error(fmt::format("TPC-TOF matched tracks ({}) require TPCTOF matches ({}) and TPCTOF tracks ({})",
                                           matchesTPCTOF.size(), tracksTPCTOF.size()));
    }
    for (unsigned i = 0; i < matchesTPCTOF.size(); i++) {
      const auto& match = matchesTPCTOF[i];
      const auto& gidx = match.getEvIdxTrack().getIndex(); // TPC (or other? but w/o ITS) track global idx (FIXME: TOF has to git rid of EvIndex stuff)
      if (isUsed(gidx)) {                                  // is TPC track already used
        continue;
      }
      const auto& trc = tracksTPCTOF[i];
      if (creator(trc, {i, GTrackID::TPCTOF}, trc.getTimeMUS().getTimeStamp(), trc.getTimeMUS().getTimeStampError())) {
        flagUsed(gidx); // flag used TPC tracks
      }
    }
  }

  // TPC only tracks
  {
    int nacc = 0, noffer = 0;
    for (unsigned i = 0; i < tracksTPC.size(); i++) {
      if (isUsed2(i, GTrackID::TPC)) { // skip used tracks
        continue;
      }
      const auto& trc = tracksTPC[i];
      if (creator(trc, {i, GTrackID::TPC}, trc.getTime0() + 0.5 * (trc.getDeltaTFwd() - trc.getDeltaTBwd()), 0.5 * (trc.getDeltaTFwd() + trc.getDeltaTBwd()))) {
        flagUsed2(i, GTrackID::TPC); // flag used TPC tracks
      }
    }
  }

  // ITS only tracks
  {
    const auto& rofrs = getITSTracksROFRecords();
    for (unsigned irof = 0; irof < rofrs.size(); irof++) {
      const auto& rofRec = rofrs[irof];
      float t0 = rofRec.getBCData().differenceInBC(startIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
      int trlim = rofRec.getFirstEntry() + rofRec.getNEntries();
      for (int it = rofRec.getFirstEntry(); it < trlim; it++) {
        if (isUsed2(it, GTrackID::ITS)) { // skip used tracks
          continue;
        }
        GTrackID gidITS(it, GTrackID::ITS);
        const auto& trc = getTrack<o2::its::TrackITS>(gidITS);
        if (creator(trc, gidITS, t0, 0.5)) {
          flagUsed2(it, GTrackID::ITS);
        }
      }
    }
  }

  // MFT only tracks
  {
    const auto& rofrs = getMFTTracksROFRecords();
    for (unsigned irof = 0; irof < rofrs.size(); irof++) {
      const auto& rofRec = rofrs[irof];
      float t0 = rofRec.getBCData().differenceInBC(startIR) * o2::constants::lhc::LHCBunchSpacingNS * 1e-3;
      int trlim = rofRec.getFirstEntry() + rofRec.getNEntries();
      for (int it = rofRec.getFirstEntry(); it < trlim; it++) {

        GTrackID gidMFT(it, GTrackID::MFT);
        const auto& trc = getTrack<o2::mft::TrackMFT>(gidMFT);
        if (creator(trc, gidMFT, t0, 0.5)) {
          flagUsed2(it, GTrackID::MFT);
        }
      }
    }
  }

  auto current_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "RecoContainer::createTracks took " << std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time).count() * 1e-6 << " CPU s.";
}

template <class T>
inline constexpr auto isITSTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::its::TrackITS>;
}

template <class T>
inline constexpr auto isMFTTrack()
{
  return std::is_same_v<std::decay_t<T>, o2::mft::TrackMFT>;
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
inline constexpr auto isTPCTRDTOFTrack()
{
  return false;
} // to be implemented

template <class T>
inline constexpr auto isITSTPCTRDTOFTrack()
{
  return false;
} // to be implemented
