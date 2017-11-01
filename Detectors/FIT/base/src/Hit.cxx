// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITBase/Hit.h"

ClassImp(o2::fit::Hit);

  using namespace o2::fit;

void Hit::PrintStream(std::ostream& stream) const
{
  stream << "FIT point:  "
         << " in detector segment " << GetDetectorID() << " at position (" << GetX() << "|" << GetY() << "|" << GetZ();
}

std::ostream& operator<<(std::ostream& stream, const Hit& p)
{
  p.PrintStream(stream);
  return stream;
}
