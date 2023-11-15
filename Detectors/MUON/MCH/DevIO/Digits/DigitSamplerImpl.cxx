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

#include "DigitSamplerImpl.h"
#include <istream>
#include "DigitFileFormat.h"
#include <map>
#include "DigitIOV0.h"
#include "DigitIOV1.h"
#include "DigitIOV2.h"
#include "DigitIOV3.h"
#include "DigitIOV4.h"
#include <memory>

namespace o2::mch::io::impl
{

std::unique_ptr<DigitSamplerImpl> createDigitSamplerImpl(int version)
{
  switch (version) {
    case 0:
      return std::make_unique<DigitSamplerV0>();
    case 1:
      return std::make_unique<DigitSamplerV1>();
    case 2:
      return std::make_unique<DigitSamplerV2>();
    case 3:
      return std::make_unique<DigitSamplerV3>();
    case 4:
      return std::make_unique<DigitSamplerV4>();
    default:
      break;
  };
  throw std::invalid_argument(fmt::format("DigitFileFormat version {} not implemented yet",
                                          version));
  return nullptr;
}

void DigitSamplerImpl::rewind(std::istream& in)
{
  in.clear();
  in.seekg(sizeof(DigitFileFormat));
}

} // namespace o2::mch::io::impl
