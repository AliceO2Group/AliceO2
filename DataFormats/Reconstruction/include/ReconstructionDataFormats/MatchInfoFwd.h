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

/// \file MatchInfoFwd.h

#ifndef ALICEO2_MATCH_INFO_MFTMCH_H
#define ALICEO2_MATCH_INFO_MFTMCH_H

#include "CommonDataFormat/InteractionRecord.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "CommonDataFormat/TimeStamp.h"

namespace o2
{
namespace dataformats
{
using timeEst = o2::dataformats::TimeStampWithError<float, float>;

class MatchInfoFwd
{
 public:
  MatchInfoFwd() = default;
  MatchInfoFwd(int32_t MCHId, int32_t MFTId, double chi2)
    : mMCHTrackID(MCHId), mMFTTrackID(MFTId), mMFTMCHMatchingChi2(chi2) {}
  ~MatchInfoFwd() = default;

  void setMFTMCHMatchingChi2(double chi2) { mMFTMCHMatchingChi2 = chi2; }
  const auto& getMFTMCHMatchingChi2() const { return mMFTMCHMatchingChi2; }

  void setMIDMatchingChi2(double chi2) { mMCHMIDMatchingChi2 = chi2; }
  const auto& getMIDMatchingChi2() const { return mMCHMIDMatchingChi2; }

  void countMFTCandidate() { mNMFTCandidates++; }
  const auto& getNMFTCandidates() const { return mNMFTCandidates; }
  void setNMFTCandidates(int n) { mNMFTCandidates = n; }

  void setCloseMatch(bool v = true) { mCloseMatch = v; }
  const auto& isCloseMatch() const { return mCloseMatch; }

  void setMFTMCHMatch(uint32_t MCHId, uint32_t MFTId, double MFTMCHMatchChi2)
  {
    mMFTTrackID = MFTId;
    mMCHTrackID = MCHId;
    mMFTMCHMatchingChi2 = MFTMCHMatchChi2;
  }

  /// get the MFT-MCH matching chi2
  double getMFTMCHMatchChi2() const { return mMFTMCHMatchingChi2; }
  /// set the MFT-MCH matching chi2
  void setMFTMCHMatchChi2(double chi2) { mMFTMCHMatchingChi2 = chi2; }

  void setMCHTrackID(int ID) { mMCHTrackID = ID; }
  const auto& getMCHTrackID() const { return mMCHTrackID; }
  void setMFTTrackID(int ID) { mMFTTrackID = ID; }
  const auto& getMFTTrackID() const { return mMFTTrackID; }

  const timeEst& getTimeMUS() const { return mTimeMUS; }
  timeEst& getTimeMUS() { return mTimeMUS; }
  void setTimeMUS(const timeEst& t) { mTimeMUS = t; }
  void setTimeMUS(float t, float te)
  {
    mTimeMUS.setTimeStamp(t);
    mTimeMUS.setTimeStampError(te);
  }

 private:
  double mMFTMCHMatchingChi2 = 1.0E308; ///< MCH-MFT Matching Chi2
  double mMCHMIDMatchingChi2 = -1.0;    ///< MCH-MID Matching Chi2
  int mMFTTrackID = -1;                 ///< Track ID of best MFT-match
  int mMCHTrackID = -1;                 ///< MCH Track ID
  int mNMFTCandidates = 0;              ///< Number of MFT candidates within search cut
  bool mCloseMatch = false;             ///< Close match = correct MFT pair tested (MC-only)
  timeEst mTimeMUS;                     ///< time estimate in ns

  ClassDefNV(MatchInfoFwd, 1);
};

} // namespace dataformats
} // namespace o2

#endif // ALICEO2_MATCH_INFO_MFTMCH_H
