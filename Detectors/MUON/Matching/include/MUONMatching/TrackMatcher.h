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

/// \file TrackMatcher.h
/// \brief Definition of a class to match MCH and MID tracks
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MUON_TRACKMATCHER_H_
#define O2_MUON_TRACKMATCHER_H_

#include <vector>

#include <gsl/span>

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"
#include "ReconstructionDataFormats/TrackMCHMID.h"

namespace o2
{
namespace muon
{

/// Class to match MCH and MID tracks
class TrackMatcher
{
  using TrackMCHMID = o2::dataformats::TrackMCHMID;

 public:
  TrackMatcher() = default;
  ~TrackMatcher() = default;

  TrackMatcher(const TrackMatcher&) = delete;
  TrackMatcher& operator=(const TrackMatcher&) = delete;
  TrackMatcher(TrackMatcher&&) = delete;
  TrackMatcher& operator=(TrackMatcher&&) = delete;

  void init();
  void match(gsl::span<const mch::ROFRecord>& mchROFs, gsl::span<const mch::TrackMCH>& mchTracks,
             gsl::span<const mid::ROFRecord>& midROFs, gsl::span<const mid::Track>& midTracks);

  /// get the MCH-MID matched tracks
  const std::vector<TrackMCHMID>& getMuons() const { return mMuons; }

 private:
  double match(const mch::TrackMCH& mchTrack, const mid::Track& midTrack);

  double mMaxChi2 = 0.;              ///< maximum chi2 to validate a MCH-MID track matching
  std::vector<TrackMCHMID> mMuons{}; ///< list of MCH-MID matched tracks
};

} // namespace muon
} // namespace o2

#endif // O2_MUON_TRACKMATCHER_H_
