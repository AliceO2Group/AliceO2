// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSelectorPID.cxx
/// \brief PID track selector class
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "AnalysisCore/TrackSelectorPID.h"

/// Constructor with PDG code initialisation
TrackSelectorPID::TrackSelectorPID(int pdg)
  : mPdg(std::abs(pdg))
{}

// TPC

/// Checks if track is OK for TPC PID.
/// \param track  track
/// \note function to be expanded
/// \return true if track is OK for TPC PID
template <typename T>
bool TrackSelectorPID::isValidTrackPIDTPC(const T& track)
{
  return track.pt() >= mPtTPCMin && track.pt() <= mPtTPCMax;
}

