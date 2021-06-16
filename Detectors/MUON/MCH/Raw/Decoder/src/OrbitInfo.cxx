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

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MCHRawDecoder/OrbitInfo.h"

namespace o2::mch
{

using RDH = o2::header::RDHAny;

OrbitInfo::OrbitInfo(gsl::span<const std::byte> rdhBuffer)
{
  auto& rdhAny = *reinterpret_cast<RDH*>(const_cast<std::byte*>(&(rdhBuffer[0])));
  auto orbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhAny);
  auto linkId = o2::raw::RDHUtils::getLinkID(rdhAny);
  auto feeId = o2::raw::RDHUtils::getFEEID(rdhAny);

  mOrbitInfo = orbit;
  mOrbitInfo += ((static_cast<uint64_t>(linkId) << 32) & 0xFF00000000);
  mOrbitInfo += ((static_cast<uint64_t>(feeId) << 40) & 0xFF0000000000);
}

bool operator==(const OrbitInfo& o1, const OrbitInfo& o2)
{
  return (o1.mOrbitInfo == o2.mOrbitInfo);
}

bool operator!=(const OrbitInfo& o1, const OrbitInfo& o2)
{
  return !(o1 == o2);
}

bool operator<(const OrbitInfo& o1, const OrbitInfo& o2)
{
  return (o1.mOrbitInfo < o2.mOrbitInfo);
}

} // namespace o2::mch
