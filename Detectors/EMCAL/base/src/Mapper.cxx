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

#include <fstream>
#include <iostream>
#include <sstream>
#include <TSystem.h>
#include "EMCALBase/Mapper.h"

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

    if (address > maxHardwareAddress) {
      throw AddressRangeException(address, maxHardwareAddress);
    }

    auto chantype = o2::emcal::intToChannelType(caloflag);

    mMapping.insert(std::pair<int, ChannelID>(static_cast<unsigned int>(address), {uint8_t(row), uint8_t(col), chantype}));
    mInverseMapping.insert(std::pair<ChannelID, int>({uint8_t(row), uint8_t(col), chantype}, static_cast<unsigned int>(address)));
  }

  mInitStatus = true;
}

Mapper::ChannelID Mapper::getChannelID(unsigned int hardawareaddress) const
{
  if (!mInitStatus) {
    throw InitStatusException();
  }
  auto res = mMapping.find(hardawareaddress);
  if (res == mMapping.end()) {
    throw AddressNotFoundException(hardawareaddress);
  }
  return res->second;
}

unsigned int Mapper::getHardwareAddress(uint8_t row, uint8_t col, ChannelType_t channeltype) const
{
  if (!mInitStatus) {
    throw InitStatusException();
  }
  ChannelID channelToFind{row, col, channeltype};
  auto found = mInverseMapping.find(channelToFind);
  if (found == mInverseMapping.end()) {
    throw ChannelNotFoundException(channelToFind);
  }
  return found->second;
}

MappingHandler::MappingHandler()
{
  const std::array<char, 2> SIDES = {{'A', 'C'}};
  const unsigned int NDDL = 2;
  for (unsigned int iside = 0; iside < 2; iside++) {
    for (unsigned int iddl = 0; iddl < NDDL; iddl++) {
      mMappings[iside * NDDL + iddl].setMapping(Form("%s/share/Detectors/EMC/files/RCU%d%c.data", gSystem->Getenv("O2_ROOT"), iddl, SIDES[iside]));
    }
  }
}

Mapper& MappingHandler::getMappingForDDL(unsigned int ddl)
{
  if (ddl >= 40) {
    throw MappingHandler::DDLInvalid(ddl);
  }
  const unsigned int NDDLSM = 2, NSIDES = 2;
  unsigned int ddlInSM = ddl % NDDLSM,
               sideID = (ddl / NDDLSM) % NSIDES;
  unsigned int mappingIndex = sideID * NDDLSM + ddlInSM;
  if (mappingIndex < 0 || mappingIndex >= mMappings.size()) {
    std::cout << "Access to invalid mapping position for ddl " << ddl << std::endl;
    throw MappingHandler::DDLInvalid(ddl);
  }
  return mMappings[mappingIndex];
}

int MappingHandler::getFEEForChannelInDDL(unsigned int ddl, unsigned int channelFEC, unsigned int branch)
{
  if (ddl >= 40) {
    throw MappingHandler::DDLInvalid(ddl);
  }
  int ddlInSupermodule = ddl % 2;
  int fecID = ddlInSupermodule ? 20 : 0;
  if (branch) {
    fecID += 10;
  }
  fecID += channelFEC;
  return fecID;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const Mapper::ChannelID& channel)
{
  stream << "Row " << static_cast<int>(channel.mRow) << ", Column " << static_cast<int>(channel.mColumn) << ", type " << o2::emcal::channelTypeToString(channel.mChannelType);
  return stream;
}
