// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSelectorPID.h
/// \brief PID track selector class
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#ifndef O2_ANALYSIS_TRACKSELECTORPID_H_
#define O2_ANALYSIS_TRACKSELECTORPID_H_

#include <TPDGCode.h>

#include "Framework/Logger.h"

/// Class for track selection using PID detectors

class TrackSelectorPID
{
 public:
  /// Default constructor
  TrackSelectorPID() = default;

  /// Standard constructor with PDG code initialisation
  explicit TrackSelectorPID(int pdg)
    : mPdg(std::abs(pdg))
  {}

  /// Default destructor
  ~TrackSelectorPID() = default;

  /// Selection status
  enum Status {
    PIDUndecided = 0,
    PIDRejected,
    PIDConditional,
    PIDAccepted
  };

  void setPDG(int pdg) { mPdg = std::abs(pdg); }

  // TPC

  /// Set pT range where TPC PID is applicable.
  void setRangePtTPC(float ptMin, float ptMax) { mPtTPCMin = ptMin; mPtTPCMax = ptMax; }

  /// Set TPC nσ range in which a track should be accepted.
  void setRangeNSigmaTPC(float nsMin, float nsMax) { mNSigmaTPCMin = nsMin; mNSigmaTPCMax = nsMax; }

  /// Checks if track is OK for TPC PID.
  /// \param track  track
  /// \return true if track is OK for TPC PID
  template <typename T>
  bool isValidTrackPIDTPC(const T& track)
  {
    auto pt = track.pt();
    return mPtTPCMin <= pt && pt <= mPtTPCMax;
  }

  /// Checks if track is compatible with given particle species hypothesis within given TPC nσ range.
  /// \param track  track
  /// \return true if track satisfies TPC PID hypothesis for given TPC nσ range
  template <typename T>
  bool isSelectedTrackPIDTPC(const T& track)
  {
    // Accept if selection is disabled via large values.
    if (mNSigmaTPCMin < -999. && mNSigmaTPCMax > 999.) {
      return true;
    }
    double nSigma = 100.;
    switch(mPdg) {
      case kPiPlus: {
        nSigma = track.tpcNSigmaPi();
        break;
      }
      case kKPlus: {
        nSigma = track.tpcNSigmaKa();
        break;
      }
      case kProton: {
        nSigma = track.tpcNSigmaPr();
        break;
      }
      default: {
        LOGF(error, "ERROR: TPC PID not implemented for PDG %d", mPdg);
        assert(false);
      }
    }
    return mNSigmaTPCMin <= nSigma && nSigma <= mNSigmaTPCMax;
  }

 private:
  uint mPdg = kPiPlus; ///< PDG code of the expected particle

  // TPC

  float mPtTPCMin = 0.; ///< minimum pT for TPC PID [GeV/c]
  float mPtTPCMax = 100.; ///< maximum pT for TPC PID [GeV/c]
  float mNSigmaTPCMin = -3.; ///< minimum number of TPC σ
  float mNSigmaTPCMax = 3.; ///< maximum number of TPC σ
};

#endif // O2_ANALYSIS_TRACKSELECTORPID_H_
