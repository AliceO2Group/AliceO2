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

#include <iostream>
#include "fairlogger/Logger.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{

namespace trd
{

uint32_t LinkRecord::getHalfChamberLinkId(uint32_t detector, uint32_t rob)
{
  int sector = (detector % (constants::NLAYER * constants::NSTACK));
  int stack = (detector % constants::NLAYER);
  int layer = ((detector % (constants::NLAYER * constants::NSTACK)) / constants::NLAYER);
  int side = rob % 2;
  LOG(debug) << "sector, stack, layer, side :" << sector << " " << stack << " " << layer << " " << side;
  return LinkRecord::getHalfChamberLinkId(sector, stack, layer, side);
}

uint32_t LinkRecord::getHalfChamberLinkId(uint32_t sector, uint32_t stack, uint32_t layer, uint32_t side)
{
  LinkRecord a;
  a.setLinkId(sector, stack, layer, side);
  return a.mLinkId;
}

void LinkRecord::setLinkId(const uint32_t sector, const uint32_t stack, const uint32_t layer, const uint32_t side)
{
  setSector(sector);
  setStack(stack);
  setLayer(layer);
  setSide(side);
  setSpare();
}

void LinkRecord::printStream(std::ostream& stream)
{
  stream << "Data for link from supermodule:" << this->getSector() << " stack:" << this->getStack() << " layer:" << this->getLayer() << "side :" << this->getSide() << ", starting from entry " << this->getFirstEntry() << " with " << this->getNumberOfObjects() << " objects";
}

std::ostream& operator<<(std::ostream& stream, LinkRecord& trg)
{
  trg.printStream(stream);
  return stream;
}

} // namespace trd
} // namespace o2
