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

/// \file TrackMCHMID.h
/// \brief Definition of the MUON track
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_TRACKMCHMID_H
#define ALICEO2_TRACKMCHMID_H

#include <utility>

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

namespace o2
{
namespace dataformats
{

/// MUON track external format
class TrackMCHMID
{
  using Time = o2::dataformats::TimeStampWithError<float, float>;

 public:
  TrackMCHMID() = default;
  TrackMCHMID(const GlobalTrackID& mchID, const GlobalTrackID& midID, const InteractionRecord& midIR, double chi2)
    : mMCHRef(mchID), mMIDRef(midID), mIR(midIR), mMatchNChi2(chi2) {}
  TrackMCHMID(uint32_t mchIdx, uint32_t midIdx, const InteractionRecord& midIR, double chi2)
    : mMCHRef(mchIdx, GlobalTrackID::MCH), mMIDRef(midIdx, GlobalTrackID::MID), mIR(midIR), mMatchNChi2(chi2) {}
  ~TrackMCHMID() = default;

  TrackMCHMID(const TrackMCHMID& track) = default;
  TrackMCHMID& operator=(const TrackMCHMID& track) = default;
  TrackMCHMID(TrackMCHMID&&) = default;
  TrackMCHMID& operator=(TrackMCHMID&&) = default;

  /// get the reference to the MCH track entry in its original container
  GlobalTrackID getMCHRef() const { return mMCHRef; }
  /// set the reference to the MCH track entry in its original container
  void setMCHRef(const GlobalTrackID& id) { mMCHRef = id; }
  /// set the reference to the MCH track entry in its original container
  void setMCHRef(uint32_t idx) { mMCHRef.set(idx, GlobalTrackID::MCH); }

  /// get the reference to the MID track entry in its original container
  GlobalTrackID getMIDRef() const { return mMIDRef; }
  /// set the reference to the MID track entry in its original container
  void setMIDRef(const GlobalTrackID& id) { mMIDRef = id; }
  /// set the reference to the MID track entry in its original container
  void setMIDRef(uint32_t idx) { mMIDRef.set(idx, GlobalTrackID::MID); }

  /// get the interaction record associated to this track
  InteractionRecord getIR() const { return mIR; }
  /// set the interaction record associated to this track
  void setIR(const InteractionRecord& ir) { mIR = ir; }

  std::pair<Time, bool> getTimeMUS(const InteractionRecord& startIR, uint32_t nOrbits = 128,
                                   bool printError = false) const;

  /// get the MCH-MID matching chi2/ndf
  double getMatchChi2OverNDF() const { return mMatchNChi2; }
  /// set the MCH-MID matching chi2/ndf
  void setMatchChi2OverNDF(double chi2) { mMatchNChi2 = chi2; }

  void print() const;

 private:
  GlobalTrackID mMCHRef{}; ///< reference to MCH track entry in its original container
  GlobalTrackID mMIDRef{}; ///< reference to MID track entry in its original container
  InteractionRecord mIR{}; ///< associated interaction record
  double mMatchNChi2 = 0.; ///< MCH-MID matching chi2/ndf

  ClassDefNV(TrackMCHMID, 1);
};

std::ostream& operator<<(std::ostream& os, const TrackMCHMID& track);

} // namespace dataformats
} // namespace o2

#endif // ALICEO2_TRACKMCHMID_H
