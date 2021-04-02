// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DigitWriterImpl.h"
#include "DigitIOV0.h"
#include "DigitIOV1.h"
#include <fmt/format.h>

namespace o2::mch::io::impl
{
std::unique_ptr<DigitWriterImpl> createDigitWriterImpl(int version)
{
  switch (version) {
    case 0:
      return std::make_unique<DigitWriterV0>();
    case 1:
      return std::make_unique<DigitWriterV1>();
    default:
      break;
  };
  throw std::invalid_argument(fmt::format("DigitFileFormat version {} not implemented yet",
                                          version));
  return nullptr;
}
} // namespace o2::mch::io::impl
