// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DigitReaderImpl.h"
#include <istream>
#include "DigitFileFormat.h"
#include <map>
#include "DigitIOV0.h"
#include "DigitIOV1.h"
#include "DigitIOV2.h"
#include "DigitIOV3.h"
#include <memory>

namespace o2::mch::io::impl
{

std::unique_ptr<DigitReaderImpl> createDigitReaderImpl(int version)
{
  switch (version) {
    case 0:
      return std::make_unique<DigitReaderV0>();
    case 1:
      return std::make_unique<DigitReaderV1>();
    case 2:
      return std::make_unique<DigitReaderV2>();
    case 3:
      return std::make_unique<DigitReaderV3>();
    default:
      break;
  };
  throw std::invalid_argument(fmt::format("DigitFileFormat version {} not implemented yet",
                                          version));
  return nullptr;
}

void DigitReaderImpl::rewind(std::istream& in)
{
  in.clear();
  in.seekg(sizeof(DigitFileFormat));
}

} // namespace o2::mch::io::impl
