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

/// \file MatchInfoTOF.cxx
/// \brief Class to store the output of the matching to TOF

#include "ReconstructionDataFormats/MatchInfoTOF.h"

using namespace o2::dataformats;

ClassImp(o2::dataformats::MatchInfoTOF);

void MatchInfoTOF::print() const
{
  printf("Match of GlobalID %s and TOF cl %d with chi2 = %.3f\n", getEvIdxTrack().getIndex().asString().c_str(),
         getTOFClIndex(), mChi2);
  mIntLT.print();
}
