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
#ifndef ALICEO2_FOCAL_PIXELLANEDATA_H
#define ALICEO2_FOCAL_PIXELLANEDATA_H

#include <array>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <string>
#include <vector>

#include <gsl/span>

namespace o2::focal
{

class PixelLanePayload
{
 public:
  PixelLanePayload() = default;
  ~PixelLanePayload() = default;

  void reset();
  void append(gsl::span<const uint8_t> payloadwords);
  void append(uint8_t word);
  gsl::span<const uint8_t> getPayload() const;

  void print(std::ostream& stream) const;

 private:
  void printWordType(uint8_t word, std::ostream& stream) const;
  std::vector<uint8_t> mPayload;
};

class PixelLaneHandler
{
 public:
  static constexpr std::size_t NLANES = 28;

  class LaneIndexException : public std::exception
  {
   public:
    LaneIndexException(int index) : std::exception(), mIndex(index), mMessage()
    {
      mMessage = "Invalid lane " + std::to_string(mIndex) + " max " + std::to_string(PixelLaneHandler::NLANES);
    }
    ~LaneIndexException() noexcept final = default;

    const char* what() const noexcept final { return mMessage.data(); }
    void print(std::ostream& stream) const;

    std::size_t getIndex() const noexcept { return mIndex; }

   private:
    std::size_t mIndex;
    std::string mMessage;
  };

  PixelLaneHandler() = default;
  ~PixelLaneHandler() = default;

  PixelLanePayload& operator[](std::size_t index) { return getLane(index); }
  const PixelLanePayload& operator[](std::size_t index) const { return getLane(index); }

  void reset();
  void resetLane(std::size_t laneID);

  PixelLanePayload& getLane(std::size_t index);
  const PixelLanePayload& getLane(std::size_t index) const;

 private:
  void handleLaneIndex(std::size_t laneIndex) const;
  std::array<PixelLanePayload, NLANES> mLaneData;
};

std::ostream& operator<<(std::ostream& stream, const PixelLaneHandler::LaneIndexException& except);
std::ostream& operator<<(std::ostream& stream, const PixelLanePayload& payload);

} // namespace o2::focal

#endif // ALICEO2_FOCAL_PIXELLANEDATA_H
