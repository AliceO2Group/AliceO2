// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// @brief implementations for placeholder class of arbitraty-version 64B-lonh RDH
// @author ruben.shahoyan@cern.ch

#include "Headers/RDHAny.h"
#include <string>
#include <cstring>

using namespace o2::header;

//_________________________________________________
/// placeholder copied from specific version
RDHAny::RDHAny(int v)
{
  if (v == 0) {
    *this = RAWDataHeader{};
  } else if (v == 6) {
    *this = RAWDataHeaderV6{};
  } else if (v == 5) {
    *this = RAWDataHeaderV5{};
  } else if (v == 3 || v == 4) {
    *this = RAWDataHeaderV4{};
  } else {
    throw std::runtime_error(std::string("unsupported RDH version ") + std::to_string(v));
  }
}

//_________________________________________________
void RDHAny::copyFrom(const void* rdh)
{
  std::memcpy(this, rdh, sizeof(RDHAny));
}
