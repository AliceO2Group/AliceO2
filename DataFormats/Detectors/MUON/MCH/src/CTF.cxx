// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <stdexcept>
#include <cstring>
#include "DataFormatsMCH/CTF.h"
#include <iostream>

namespace o2::mch
{
std::ostream& operator<<(std::ostream& os, const CTFHeader& ctf)
{
  os << fmt::format("nROFS {} nDigits {} firstOrbit {} firstBC {}",
                    ctf.nROFs, ctf.nDigits, ctf.firstOrbit, ctf.firstBC);
  return os;
}
} // namespace o2::mch
