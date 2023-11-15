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
#ifndef ALICEO2_FOCAL_PADMAPPER_H
#define ALICEO2_FOCAL_PADMAPPER_H

#include <array>
#include <exception>
#include <iosfwd>
#include <string>
#include <tuple>

namespace o2::focal
{

class PadMapper
{
 public:
  static constexpr std::size_t NCOLUMN = 8;
  static constexpr std::size_t NROW = 9;
  static constexpr std::size_t NCHANNELS = NCOLUMN * NROW;

  class PositionException : public std::exception
  {
   public:
    PositionException(unsigned int column, unsigned int row) : mColumn(column), mRow(row), mMessage()
    {
      mMessage = "Invalid pad position: col (" + std::to_string(mColumn) + "), row (" + std::to_string(mRow);
    }
    ~PositionException() noexcept final = default;

    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    unsigned int getColumn() const noexcept { return mColumn; }
    unsigned int getRow() const noexcept { return mRow; }
    void print(std::ostream& stream) const;

   private:
    unsigned int mColumn;
    unsigned int mRow;
    std::string mMessage;
  };

  class ChannelIDException : public std::exception
  {
   public:
    ChannelIDException(unsigned int channelID) : mChannelID(channelID), mMessage()
    {
      mMessage = "Invalid channelID: " + std::to_string(mChannelID);
    }
    ~ChannelIDException() noexcept final = default;

    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    unsigned int getChannelID() const { return mChannelID; }
    void print(std::ostream& stream) const;

   private:
    unsigned int mChannelID;
    std::string mMessage;
  };
  PadMapper();
  ~PadMapper() = default;

  std::tuple<unsigned int, unsigned int> getRowColFromChannelID(unsigned int channelID) const;
  unsigned int getRow(unsigned int channelID) const;
  unsigned int getColumn(unsigned int channelID) const;

  unsigned int getChannelID(unsigned int col, unsigned int row) const;

 private:
  static constexpr unsigned int mMapping[NCOLUMN][NROW] = { // map of channels in XY
    {43, 41, 53, 49, 58, 54, 66, 68, 69},
    {39, 37, 47, 51, 56, 62, 60, 64, 70},
    {38, 42, 45, 46, 50, 59, 55, 65, 71},
    {44, 40, 36, 48, 52, 61, 57, 67, 63},
    {8, 4, 0, 14, 16, 19, 23, 31, 27},
    {2, 6, 11, 10, 17, 21, 25, 33, 35},
    {1, 3, 9, 15, 22, 24, 26, 28, 34},
    {7, 5, 12, 13, 18, 20, 29, 30, 32}};

  void initInverseMapping();

  std::array<std::tuple<unsigned int, unsigned int>, NCHANNELS> mInverseMapping; ///< Inverse mapping (Channel ID -> Col/Row)
};

std::ostream& operator<<(std::ostream& stream, const PadMapper::PositionException& except);
std::ostream& operator<<(std::ostream& stream, const PadMapper::ChannelIDException& except);

} // namespace o2::focal

#endif // ALICEO2_FOCAL_PADMAPPER_H