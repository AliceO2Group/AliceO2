// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"

#include <cstdint>
#include <chrono>

namespace o2
{
namespace framework
{

uint64_t DataProcessingHeader::getCreationTime()
{
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

constexpr o2::header::HeaderType DataProcessingHeader::sHeaderType;

} // namespace framework
} // namespace o2
