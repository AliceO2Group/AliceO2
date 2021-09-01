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
/// \brief Implementation of the MUON track
///
/// \author Philippe Pillot, Subatech

#include "ReconstructionDataFormats/TrackMCHMID.h"

#include <iostream>

namespace o2
{
namespace dataformats
{

//__________________________________________________________________________
/// write the content of the track to the output stream
std::ostream& operator<<(std::ostream& os, const o2::dataformats::TrackMCHMID& track)
{
  os << track.getMCHRef() << " + " << track.getMIDRef() << " = "
     << track.getIR() << " matching chi2/NDF: " << track.getMatchChi2OverNDF();
  return os;
}

//__________________________________________________________________________
/// write the content of the track to the standard output
void TrackMCHMID::print() const
{
  std::cout << *this << std::endl;
}

} // namespace dataformats
} // namespace o2
