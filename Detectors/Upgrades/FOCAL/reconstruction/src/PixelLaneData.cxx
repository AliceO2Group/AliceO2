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
#include <FOCALReconstruction/PixelLaneData.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iomanip>

using namespace o2::focal;

void PixelLaneHandler::reset()
{
  for (auto& lane : mLaneData) {
    lane.reset();
  }
}

void PixelLaneHandler::resetLane(std::size_t laneID)
{
  handleLaneIndex(laneID);
  mLaneData[laneID].reset();
}

PixelLanePayload& PixelLaneHandler::getLane(std::size_t index)
{
  handleLaneIndex(index);
  return mLaneData[index];
}

const PixelLanePayload& PixelLaneHandler::getLane(std::size_t index) const
{
  handleLaneIndex(index);
  return mLaneData[index];
}

void PixelLaneHandler::handleLaneIndex(std::size_t laneIndex) const
{
  if (laneIndex >= PixelLaneHandler::NLANES) {
    throw PixelLaneHandler::LaneIndexException(laneIndex);
  }
}

void PixelLanePayload::reset()
{
  mPayload.clear();
}

void PixelLanePayload::append(gsl::span<const uint8_t> payloadwords)
{
  std::copy(payloadwords.begin(), payloadwords.end(), std::back_inserter(mPayload));
}

void PixelLanePayload::append(uint8_t word)
{
  mPayload.emplace_back(word);
}

gsl::span<const uint8_t> PixelLanePayload::getPayload() const
{
  return mPayload;
}

void PixelLanePayload::print(std::ostream& stream) const
{
  stream << "Next lane with " << mPayload.size() << " words:";
  for (const auto& word : mPayload) {
    stream << " 0x" << std::hex << static_cast<int>(word) << std::dec;
  }
}

void PixelLaneHandler::LaneIndexException::print(std::ostream& stream) const
{
  stream << mMessage;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PixelLaneHandler::LaneIndexException& except)
{
  except.print(stream);
  return stream;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PixelLanePayload& payload)
{
  payload.print(stream);
  return stream;
}