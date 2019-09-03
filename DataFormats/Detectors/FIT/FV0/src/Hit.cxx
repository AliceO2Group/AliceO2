// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Hit.cxx
/// \brief Implementation of the Hit class

#include "DataFormatsFV0/Hit.h"

#include <iostream>

ClassImp(o2::fv0::Hit);

namespace o2
{
  namespace fv0
  {

  void Hit::Print(const Option_t* opt) const
  {
    printf(
      "Det: %5d Track: %6d E.loss: %.3e P: %+.3e %+.3e %+.3e\n"
      "PosIn: %+.3e %+.3e %+.3e PosOut: %+.3e %+.3e %+.3e\n",
      GetDetectorID(), GetTrackID(), GetEnergyLoss(), GetPx(), GetPy(), GetPz(),
      GetStartX(), GetStartY(), GetStartZ(), GetX(), GetY(), GetZ());
  }

  } // namespace fv0
} // namespace o2
