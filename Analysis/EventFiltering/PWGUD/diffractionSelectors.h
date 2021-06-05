// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ANALYSIS_DIFFRACTION_SELECTOR_H_
#define O2_ANALYSIS_DIFFRACTION_SELECTOR_H_

#include "Framework/DataTypes.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/PID/PIDResponse.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

template <typename TC>
bool hasGoodPID(cutHolder diffCuts, TC track);

// add here Selectors for different types of diffractive events
// Selector for Double Gap events
struct DGSelector {
 public:
  DGSelector() = default;

  // Function to check if collisions passes filter
  template <typename CC, typename BC, typename BCs, typename TCs, typename MUs>
  bool IsSelected(cutHolder diffCuts, CC const& collision, BC& bc, BCs& bcs, TCs& tracks, MUs& muons)
  {
    LOGF(info, "Collision %f BC %i", collision.collisionTime(), bc.globalBC());
    LOGF(info, "Number of close BCs: %i", bcs.size());
    LOGF(info, "Number of tracks: %i", tracks.size());

    // characteristics of collision and nominal bc
    // Decisions which do NOT need pre- post-pileup protection

    // Number of tracks
    //if (tracks.size() < diffCuts.minNTracks() || tracks.size() > diffCuts.maxNTracks()) {
    //  return false;
    //}

    // all tracks must be global tracks
    // ATTENTION: isGlobalTrack is created by o2-analysis-trackselection
    //            The trackselection currently has fixed settings and can not be configured with
    //            command line options! This is not what we want!
    LOGF(info, "Track types");
    for (auto& track : tracks) {
      LOGF(info, "  : %i / %i / %i / %i", track.trackType(), track.isGlobalTrack(), track.isGlobalTrackSDD(), track.itsClusterMap());
    }

    // tracks with TOF hits
    // ATTENTION: in Run 2 converted data hasTOF is often true although the tofSignal
    //            has a dummy value (99998.) This will be corrected in new versions
    //            of the data - check this!
    LOGF(info, "TOF");
    int nTracksWithTOFHit = 0;
    for (auto& track : tracks) {
      LOGF(info, "  [%i] signal %f / chi2: %f", track.hasTOF(), track.tofSignal(), track.tofChi2());
    }
    LOGF(info, "Tracks with TOF hit %i", nTracksWithTOFHit);

    // only tracks with good PID
    for (auto& track : tracks) {
      auto goodPID = hasGoodPID(diffCuts, track);
    }

    // Decisions which need past-future protection
    // This applies to 'slow' detectors
    // loop over all selected BCs
    LOGF(info, "Number of close BCs: %i", bcs.size());
    for (auto& bc : bcs) {
      LOGF(info, "BC %i / %i", bc.globalBC(), bc.triggerMask());

      // check no activity in FT0
      LOGF(info, "FT0 %i", collision.foundFT0());
      if (bc.has_ft0()) {
        LOGF(info, "  %f / %f / %f / %f",
             bc.ft0().timeA(), bc.ft0().amplitudeA()[0],
             bc.ft0().timeC(), bc.ft0().amplitudeC()[0]);
      }

      // check no activity in FV0-A
      LOGF(info, "FV0-A");
      if (bc.has_fv0a()) {
        LOGF(info, "  %f / %f", bc.fv0a().time(), bc.fv0a().amplitude()[0]);
      }

      // check no activity in FV0-C
      // ATTENTION: not available in Run 3 data
      //LOGF(info, "FV0-C");
      //if (bc.has_fv0c()) {
      //  LOGF(info, "  %f / %f", bc.fv0c().time(), bc.fv0c().amplitude()[0]);
      //}

      // check no activity in FDD (AD Run 2)
      LOGF(info, "FDD");
      if (bc.has_fdd()) {
        LOGF(info, "  %f / %f / %f / %f",
             bc.fdd().timeA(), bc.fdd().amplitudeA()[0],
             bc.fdd().timeC(), bc.fdd().amplitudeC()[0]);
      }

      // check no activity in ZDC
      LOGF(info, "ZDC");
      if (bc.has_zdc()) {
        LOGF(info, "  %f / %f / %f / %f",
             bc.zdc().timeZEM1(), bc.zdc().energyZEM1(),
             bc.zdc().timeZEM2(), bc.zdc().energyZEM2());
      }

      // check no activity in muon arm
      LOGF(info, "Muons %i", muons.size());
      for (auto& muon : muons) {
        LOGF(info, "  %i / %f / %f / %f", muon.trackType(), muon.eta(), muon.pt(), muon.p());
      }
    }

    // ATTENTION: currently all events are selected
    return true;
  };
};

// function to check if track provides good PID information
// Checks the nSigma for any particle assumption to be within limits.
template <typename TC>
bool hasGoodPID(cutHolder diffCuts, TC track)
{
  // El, Mu, Pi, Ka, and Pr are considered
  // at least one nSigma must be within set limits
  LOGF(info, "TPC PID %f / %f / %f / %f / %f",
       track.tpcNSigmaEl(),
       track.tpcNSigmaMu(),
       track.tpcNSigmaPi(),
       track.tpcNSigmaKa(),
       track.tpcNSigmaPr());
  if (TMath::Abs(track.tpcNSigmaEl()) < diffCuts.maxnSigmaTPC()) {
    return true;
  }
  if (TMath::Abs(track.tpcNSigmaMu()) < diffCuts.maxnSigmaTPC()) {
    return true;
  }
  if (TMath::Abs(track.tpcNSigmaPi()) < diffCuts.maxnSigmaTPC()) {
    return true;
  }
  if (TMath::Abs(track.tpcNSigmaKa()) < diffCuts.maxnSigmaTPC()) {
    return true;
  }
  if (TMath::Abs(track.tpcNSigmaPr()) < diffCuts.maxnSigmaTPC()) {
    return true;
  }

  if (track.hasTOF()) {
    LOGF(info, "TOF PID %f / %f / %f / %f / %f",
         track.tofNSigmaEl(),
         track.tofNSigmaMu(),
         track.tofNSigmaPi(),
         track.tofNSigmaKa(),
         track.tofNSigmaPr());
    if (TMath::Abs(track.tofNSigmaEl()) < diffCuts.maxnSigmaTOF()) {
      return true;
    }
    if (TMath::Abs(track.tofNSigmaMu()) < diffCuts.maxnSigmaTOF()) {
      return true;
    }
    if (TMath::Abs(track.tofNSigmaPi()) < diffCuts.maxnSigmaTOF()) {
      return true;
    }
    if (TMath::Abs(track.tofNSigmaKa()) < diffCuts.maxnSigmaTOF()) {
      return true;
    }
    if (TMath::Abs(track.tofNSigmaPr()) < diffCuts.maxnSigmaTOF()) {
      return true;
    }
  }
  return false;
}

#endif // O2_ANALYSIS_DIFFRACTION_SELECTOR_H_
