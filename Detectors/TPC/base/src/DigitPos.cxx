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

#include "TPCBase/DigitPos.h"
#include "TPCBase/Mapper.h"

namespace o2
{
namespace tpc
{

PadPos DigitPos::getGlobalPadPos() const
{
  const Mapper& mapper = Mapper::instance();
  const PadRegionInfo& p = mapper.getPadRegionInfo(mCRU.region());

  return PadPos(mPadPos.getRow() + p.getGlobalRowOffset(), mPadPos.getPad());
}

PadSecPos DigitPos::getPadSecPos() const
{
  return PadSecPos(mCRU.sector(), getGlobalPadPos());
}

} // namespace tpc
} // namespace o2
