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

#include "DataFormatsMCH/ROFRecord.h"
#include <fmt/format.h>
#include <iostream>
#include <stdexcept>

namespace o2::mch
{
std::ostream& operator<<(std::ostream& os, const ROFRecord& rof)
{
  os << fmt::format("{} FirstIdx: {:5d} LastIdx: {:5d} Width: {:2d} BCs",
                    rof.getBCData().asString(), rof.getFirstIdx(), rof.getLastIdx(),
                    rof.getBCWidth());
  return os;
}
void ROFRecord::setBCWidth(int bcWidth)
{
  if (bcWidth < 4) {
    throw std::invalid_argument(fmt::format("bcWidth must be strictly greater than 4 bunch-crossings"));
  }
}

} // namespace o2::mch
