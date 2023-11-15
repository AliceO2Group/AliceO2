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

#include <fairlogger/Logger.h>
#include <FOCALReconstruction/PixelMapper.h>

using namespace o2::focal;

PixelMapper::PixelMapper(PixelMapper::MappingType_t mappingtype) : mMappingType(mappingtype)
{
  switch (mappingtype) {
    case MappingType_t::MAPPING_IB:
      mMappingFile = Form("%s/share/Detectors/FOC/files/mapping_ib.data", gSystem->Getenv("O2_ROOT"));
      break;
    case MappingType_t::MAPPING_OB:
      mMappingFile = Form("%s/share/Detectors/FOC/files/mapping_ob.data", gSystem->Getenv("O2_ROOT"));
    default:
      break;
  };

  if (mMappingFile.length()) {
    init();
  }
}

void PixelMapper::init()
{
  if (gSystem->AccessPathName(mMappingFile.data())) {
    throw MappingNotSetException();
  }
  LOG(debug) << "Reading pixel mapping from file: " << mMappingFile;
  mMapping.clear();
  std::ifstream reader(mMappingFile);
  std::string buffer;
  while (std::getline(reader, buffer)) {
    auto delimiter = buffer.find("//");
    std::string data = buffer;
    if (delimiter == 0) {
      // whole line commented out
      continue;
    }
    if (delimiter != std::string::npos) {
      data = buffer.substr(0, delimiter);
    }
    LOG(debug) << "Processing line: " << data;
    std::stringstream decoder(data);
    std::string linebuffer;
    std::vector<int> identifiers;
    while (std::getline(decoder, linebuffer, ',')) {
      identifiers.push_back(std::stoi(linebuffer));
    }
    if (identifiers.size() < 8) {
      LOG(error) << "Chip coordinates not fully defined (" << data << "), skipping ...";
    }
    ChipIdentifier nextIdentifier;
    nextIdentifier.mFEEID = identifiers[0];
    nextIdentifier.mLaneID = identifiers[1];
    nextIdentifier.mChipID = identifiers[2];
    ChipPosition nextPosition;
    nextPosition.mLayer = identifiers[3];
    nextPosition.mColumn = identifiers[4];
    nextPosition.mRow = identifiers[5];
    nextPosition.mInvertColumn = (identifiers[6] == 1 ? true : false);
    nextPosition.mInvertRow = (identifiers[7] == 1 ? true : false);
    LOG(debug) << "Inserting chip: (FEE " << nextIdentifier.mFEEID << ", Lane " << nextIdentifier.mLaneID << ", Chip " << nextIdentifier.mChipID << ") -> (Layer " << nextPosition.mLayer << ", Col " << nextPosition.mColumn << ", Row " << nextPosition.mRow << ", Inv Col " << (nextPosition.mInvertColumn ? "yes" : "no") << ", Inv Row " << (nextPosition.mInvertRow ? "yes" : "no") << ")";
    if (mMapping.find(nextIdentifier) != mMapping.end()) {
      LOG(error) << "Chip with FEE" << nextIdentifier.mFEEID << ", Lane " << nextIdentifier.mLaneID << ", Chip " << nextIdentifier.mChipID << " already present, not overwriting ...";
      continue;
    }
    mMapping.insert({nextIdentifier, nextPosition});
    if (nextPosition.mRow + 1 > mNumberOfRows) {
      mNumberOfRows = nextPosition.mRow + 1;
    }
    if (nextPosition.mColumn + 1 > mNumberOfColumns) {
      mNumberOfColumns = nextPosition.mColumn + 1;
    }
  }
  LOG(info) << "Pixel Mapper: Found " << mMapping.size() << " chips, in " << mNumberOfColumns << " colums and " << mNumberOfRows << " rows";
  reader.close();
}

PixelMapper::ChipPosition PixelMapper::getPosition(unsigned int feeID, unsigned int laneID, unsigned int chipID) const
{
  auto cardindex = feeID & 0x00FF;
  checkInitialized();
  ChipIdentifier identifier{cardindex, laneID, chipID};
  auto found = mMapping.find(identifier);
  if (found == mMapping.end()) {
    throw InvalidChipException(identifier);
  }
  return found->second;
}

void PixelMapper::checkInitialized() const
{
  if (!mMapping.size()) {
    throw UninitException();
  }
}

void PixelMapper::InvalidChipException::print(std::ostream& stream) const
{
  stream << mMessage;
}

void PixelMapper::UninitException::print(std::ostream& stream) const
{
  stream << what();
}

void PixelMapper::MappingNotSetException::print(std::ostream& stream) const
{
  stream << what();
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PixelMapper::InvalidChipException& error)
{
  error.print(stream);
  return stream;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PixelMapper::UninitException& error)
{
  error.print(stream);
  return stream;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PixelMapper::MappingNotSetException& error)
{
  error.print(stream);
  return stream;
}