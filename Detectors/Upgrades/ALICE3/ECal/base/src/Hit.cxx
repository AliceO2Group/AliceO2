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

/// \file Hit.cxx
/// \brief MC hit class to store energy loss per cell and per superparent
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#include <ECalBase/Hit.h>

ClassImp(o2::ecal::Hit);

using namespace o2::ecal;

bool Hit::operator<(const Hit& rhs) const
{
  if (GetTrackID() != rhs.GetTrackID()) {
    return GetTrackID() < rhs.GetTrackID();
  }
  return GetCellID() < rhs.GetCellID();
}

bool Hit::operator==(const Hit& rhs) const
{
  return (GetCellID() == rhs.GetCellID()) && (GetTrackID() == rhs.GetTrackID());
}
