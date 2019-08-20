// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <fstream>
#include <iostream>
#include <sstream>
#include "EMCALReconstruction/Mapper.h"

using namespace o2::emcal;

Mapper::Mapper(const std::string_view filename) : mMapping(),
                                                  mInverseMapping()
{
  init(filename); // will throw exceptions in case the initialization from file failed
}

void Mapper::setMapping(const std::string_view inputfile)
{
  mMapping.clear();
  mInverseMapping.clear();
  mInitStatus = false;
  init(inputfile);
}

void Mapper::init(const std::string_view filename)
{
  std::ifstream reader(filename.data());
  std::string tmp;

  int maxHardwareAddress(-1);
  try {
    std::getline(reader, tmp); // max. number of entries - no longer needed
    std::getline(reader, tmp);
    maxHardwareAddress = std::stoi(tmp);
  } catch (std::invalid_argument& e) {
    throw FileFormatException(e.what());
  } catch (std::out_of_range& e) {
    throw FileFormatException(e.what());
  } catch (std::iostream::failure& e) {
    throw FileFormatException(e.what());
  }

  // loop over channels
  int address, row, col, caloflag;
  while (getline(reader, tmp)) {
    std::stringstream addressmapping(tmp);

    try {
      addressmapping >> address >> row >> col >> caloflag;
    } catch (std::iostream::failure& e) {
      throw FileFormatException(e.what());
    }

    if (address > maxHardwareAddress)
      throw AddressRangeException(address, maxHardwareAddress);

    auto chantype = o2::emcal::intToChannelType(caloflag);

    mMapping.insert(std::pair<int, ChannelID>(address, {uint8_t(row), uint8_t(col), chantype}));
    mInverseMapping.insert(std::pair<ChannelID, int>({uint8_t(row), uint8_t(col), chantype}, address));
  }

  mInitStatus = true;
}

Mapper::ChannelID Mapper::getChannelID(int hardawareaddress) const
{
  if (!mInitStatus)
    throw InitStatusException();
  auto res = mMapping.find(hardawareaddress);
  if (res == mMapping.end())
    throw AddressNotFoundException(hardawareaddress);
  return res->second;
}

int Mapper::getHardwareAddress(int row, int col, ChannelType_t channeltype) const
{
  if (!mInitStatus)
    throw InitStatusException();
  ChannelID channelToFind{uint8_t(row), uint8_t(col), channeltype};
  auto found = mInverseMapping.find(channelToFind);
  if (found == mInverseMapping.end())
    throw ChannelNotFoundException(channelToFind);
  return found->second;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const Mapper::ChannelID& channel)
{
  stream << "Row " << channel.mRow << ", Column " << channel.mColumn << ", type " << o2::emcal::channelTypeToString(channel.mChannelType);
  return stream;
}