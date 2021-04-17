// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsMCH/ROFRecord.h"
#include <fmt/format.h>
#include <iostream>

namespace o2::mch
{
std::ostream& operator<<(std::ostream& os, const ROFRecord& rof)
{
  os << fmt::format("{} FirstIdx: {:5d} LastIdx: {:5d}",
                    rof.getBCData().asString(), rof.getFirstIdx(), rof.getLastIdx());
  return os;
}
} // namespace o2::mch
