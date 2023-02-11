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

#include "DigitSinkImpl.h"
#include "DigitIOV0.h"
#include "DigitIOV1.h"
#include "DigitIOV2.h"
#include "DigitIOV3.h"
#include "DigitIOV4.h"
#include <fmt/format.h>

namespace o2::mch::io::impl
{
std::unique_ptr<DigitSinkImpl> createDigitSinkImpl(int version)
{
  switch (version) {
    case 0:
      return std::make_unique<DigitSinkV0>();
    case 1:
      return std::make_unique<DigitSinkV1>();
    case 2:
      return std::make_unique<DigitSinkV2>();
    case 3:
      return std::make_unique<DigitSinkV3>();
    case 4:
      return std::make_unique<DigitSinkV4>();
    default:
      break;
  };
  throw std::invalid_argument(fmt::format("DigitFileFormat version {} not implemented yet",
                                          version));
  return nullptr;
}

} // namespace o2::mch::io::impl
