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
  {
  }

  /// Default destructor
  ~TrackSelectorPID() = default;

  /// Selection status
  enum Status {
    PIDNotApplicable = 0,
    PIDRejected,
    PIDConditional,
    PIDAccepted
  };

  void setPDG(int pdg) { mPdg = std::abs(pdg); }

  // TPC

  /// Set pT range where TPC PID is applicable.
  void setRangePtTPC(float ptMin, float ptMax)
  {
    mPtTPCMin = ptMin;
    mPtTPCMax = ptMax;
  }

  /// Set TPC nσ range in which a track should be accepted.
  void setRangeNSigmaTPC(float nsMin, float nsMax)
  {
    mNSigmaTPCMin = nsMin;
    mNSigmaTPCMax = nsMax;
  }

  /// Set TPC nσ range in which a track should be conditionally accepted if combined with TOF.
  void setRangeNSigmaTPCCondTOF(float nsMin, float nsMax)
  {
    mNSigmaTPCMinCondTOF = nsMin;
    mNSigmaTPCMaxCondTOF = nsMax;
  }

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
  /// \param conditionalTOF  variable to store the result of selection with looser cuts for conditional accepting of track if combined with TOF
  /// \return true if track satisfies TPC PID hypothesis for given TPC nσ range
  template <typename T>
  bool isSelectedTrackPIDTPC(const T& track, bool& conditionalTOF)
  {
    // Accept if selection is disabled via large values.
    if (mNSigmaTPCMin < -999. && mNSigmaTPCMax > 999.) {
      return true;
    }

    // Get nσ for a given particle hypothesis.
    double nSigma = 100.;
    switch (mPdg) {
      case kElectron: {
        nSigma = track.tpcNSigmaEl();
        break;
      }
      case kMuonMinus: {
        nSigma = track.tpcNSigmaMu();
        break;
      }
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

    if (mNSigmaTPCMinCondTOF < -999. && mNSigmaTPCMaxCondTOF > 999.) {
      conditionalTOF = true;
    } else {
      conditionalTOF = mNSigmaTPCMinCondTOF <= nSigma && nSigma <= mNSigmaTPCMaxCondTOF;
    }
    return mNSigmaTPCMin <= nSigma && nSigma <= mNSigmaTPCMax;
  }

  /// Returns status of TPC PID selection for a given track.
  /// \param track  track
  /// \return TPC selection status (see TrackSelectorPID::Status)
  template <typename T>
  int getStatusTrackPIDTPC(const T& track)
  {
    if (isValidTrackPIDTPC(track)) {
      bool condTOF = false;
      if (isSelectedTrackPIDTPC(track, condTOF)) {
        return Status::PIDAccepted; // accepted
      } else if (condTOF) {
        return Status::PIDConditional; // potential to be accepted if combined with TOF
      } else {
        return Status::PIDRejected; // rejected
      }
    } else {
      return Status::PIDNotApplicable; // PID not applicable
    }
  }

  // TOF

  /// Set pT range where TOF PID is applicable.
  void setRangePtTOF(float ptMin, float ptMax)
  {
    mPtTOFMin = ptMin;
    mPtTOFMax = ptMax;
  }

  /// Set TOF nσ range in which a track should be accepted.
  void setRangeNSigmaTOF(float nsMin, float nsMax)
  {
    mNSigmaTOFMin = nsMin;
    mNSigmaTOFMax = nsMax;
  }

  /// Set TOF nσ range in which a track should be conditionally accepted if combined with TPC.
  void setRangeNSigmaTOFCondTPC(float nsMin, float nsMax)
  {
    mNSigmaTOFMinCondTPC = nsMin;
    mNSigmaTOFMaxCondTPC = nsMax;
  }

  /// Checks if track is OK for TOF PID.
  /// \param track  track
  /// \return true if track is OK for TOF PID
  template <typename T>
  bool isValidTrackPIDTOF(const T& track)
  {
    auto pt = track.pt();
    return mPtTOFMin <= pt && pt <= mPtTOFMax;
  }

  /// Checks if track is compatible with given particle species hypothesis within given TOF nσ range.
  /// \param track  track
  /// \param conditionalTPC  variable to store the result of selection with looser cuts for conditional accepting of track if combined with TPC
  /// \return true if track satisfies TOF PID hypothesis for given TOF nσ range
  template <typename T>
  bool isSelectedTrackPIDTOF(const T& track, bool& conditionalTPC)
  {
    // Accept if selection is disabled via large values.
    if (mNSigmaTOFMin < -999. && mNSigmaTOFMax > 999.) {
      return true;
    }

    // Get nσ for a given particle hypothesis.
    double nSigma = 100.;
    switch (mPdg) {
      case kElectron: {
        nSigma = track.tofNSigmaEl();
        break;
      }
      case kMuonMinus: {
        nSigma = track.tofNSigmaMu();
        break;
      }
      case kPiPlus: {
        nSigma = track.tofNSigmaPi();
        break;
      }
      case kKPlus: {
        nSigma = track.tofNSigmaKa();
        break;
      }
      case kProton: {
        nSigma = track.tofNSigmaPr();
        break;
      }
      default: {
        LOGF(error, "ERROR: TOF PID not implemented for PDG %d", mPdg);
        assert(false);
      }
    }

    if (mNSigmaTOFMinCondTPC < -999. && mNSigmaTOFMaxCondTPC > 999.) {
      conditionalTPC = true;
    } else {
      conditionalTPC = mNSigmaTOFMinCondTPC <= nSigma && nSigma <= mNSigmaTOFMaxCondTPC;
    }
    return mNSigmaTOFMin <= nSigma && nSigma <= mNSigmaTOFMax;
  }

  /// Returns status of TOF PID selection for a given track.
  /// \param track  track
  /// \return TOF selection status (see TrackSelectorPID::Status)
  template <typename T>
  int getStatusTrackPIDTOF(const T& track)
  {
    if (isValidTrackPIDTOF(track)) {
      bool condTPC = false;
      if (isSelectedTrackPIDTOF(track, condTPC)) {
        return Status::PIDAccepted; // accepted
      } else if (condTPC) {
        return Status::PIDConditional; // potential to be accepted if combined with TPC
      } else {
        return Status::PIDRejected; // rejected
      }
    } else {
      return Status::PIDNotApplicable; // PID not applicable
    }
  }

  /// Returns status of combined PID selection for a given track.
  /// \param track  track
  /// \return combined-selection status (see TrackSelectorPID::Status)
  template <typename T>
  int getStatusTrackPIDAll(const T& track)
  {
    int statusTPC = getStatusTrackPIDTPC(track);
    int statusTOF = getStatusTrackPIDTOF(track);

    if (statusTPC == Status::PIDAccepted || statusTOF == Status::PIDAccepted) {
      return Status::PIDAccepted; // what if we have Accepted for one and Rejected for the other?
    }
    if (statusTPC == Status::PIDConditional && statusTOF == Status::PIDConditional) {
      return Status::PIDAccepted;
    }
    if (statusTPC == Status::PIDRejected || statusTOF == Status::PIDRejected) {
      return Status::PIDRejected;
    }
    return Status::PIDNotApplicable; // (NotApplicable for one detector) and (NotApplicable or Conditional for the other)
  }

 private:
  uint mPdg = kPiPlus; ///< PDG code of the expected particle

  // TPC
  float mPtTPCMin = 0.;                ///< minimum pT for TPC PID [GeV/c]
  float mPtTPCMax = 100.;              ///< maximum pT for TPC PID [GeV/c]
  float mNSigmaTPCMin = -3.;           ///< minimum number of TPC σ
  float mNSigmaTPCMax = 3.;            ///< maximum number of TPC σ
  float mNSigmaTPCMinCondTOF = -1000.; ///< minimum number of TPC σ if combined with TOF
  float mNSigmaTPCMaxCondTOF = 1000.;  ///< maximum number of TPC σ if combined with TOF

  // TOF
  float mPtTOFMin = 0.;                ///< minimum pT for TOF PID [GeV/c]
  float mPtTOFMax = 100.;              ///< maximum pT for TOF PID [GeV/c]
  float mNSigmaTOFMin = -3.;           ///< minimum number of TOF σ
  float mNSigmaTOFMax = 3.;            ///< maximum number of TOF σ
  float mNSigmaTOFMinCondTPC = -1000.; ///< minimum number of TOF σ if combined with TPC
  float mNSigmaTOFMaxCondTPC = 1000.;  ///< maximum number of TOF σ if combined with TPC
};

#endif // O2_ANALYSIS_TRACKSELECTORPID_H_
