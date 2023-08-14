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
#ifndef ALICEO2_FOCAL_PIXELMAPPER_H
#define ALICEO2_FOCAL_PIXELMAPPER_H

#include <cstdio>
#include <array>
#include <exception>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <boost/container_hash/hash.hpp>
#include <DataFormatsFOCAL/PixelChip.h>
#include <Rtypes.h>

namespace o2::focal
{

class PixelMapper
{
 public:
  enum class MappingType_t {
    MAPPING_IB,
    MAPPING_OB,
    MAPPING_UNKNOWN
  };

  struct ChipIdentifier {
    unsigned int mFEEID;
    unsigned int mLaneID;
    unsigned int mChipID;

    bool operator==(const ChipIdentifier& other) const { return mFEEID == other.mFEEID && mLaneID == other.mLaneID && mChipID == other.mChipID; }
  };
  struct ChipPosition {
    unsigned int mColumn;
    unsigned int mRow;
    unsigned int mLayer;
    bool mInvertColumn;
    bool mInvertRow;

    bool operator==(const ChipPosition& other) const { return mLayer == other.mLayer && mColumn == other.mColumn && mRow == other.mRow; }
  };
  struct ChipIdentifierHasher {

    /// \brief Functor implementation
    /// \param s ChipID for which to determine a hash value
    /// \return hash value for channel ID
    size_t operator()(const ChipIdentifier& s) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, s.mFEEID);
      boost::hash_combine(seed, s.mChipID);
      boost::hash_combine(seed, s.mLaneID);
      return seed;
    }
  };

  class InvalidChipException : public std::exception
  {
   public:
    InvalidChipException(PixelMapper::ChipIdentifier& identifier) : mIdentifier(identifier), mMessage()
    {
      mMessage = "Invalid chip identifier: FEE " + std::to_string(mIdentifier.mFEEID) + ", lane " + std::to_string(mIdentifier.mLaneID) + ", chip " + std::to_string(mIdentifier.mChipID);
    }
    ~InvalidChipException() noexcept final = default;

    const char* what() const noexcept final { return mMessage.data(); }
    const PixelMapper::ChipIdentifier& getIdentifier() const { return mIdentifier; }
    unsigned int getLane() const noexcept { return mIdentifier.mLaneID; }
    unsigned int getChipID() const noexcept { return mIdentifier.mChipID; }
    unsigned int getFEEID() const noexcept { return mIdentifier.mFEEID; }
    void print(std::ostream& stream) const;

   private:
    PixelMapper::ChipIdentifier mIdentifier;
    std::string mMessage;
  };

  class UninitException : public std::exception
  {
   public:
    UninitException() = default;
    ~UninitException() noexcept final = default;

    const char* what() const noexcept final { return "Mapping is not initalized"; }
    void print(std::ostream& stream) const;
  };

  class MappingNotSetException : public std::exception
  {
   public:
    MappingNotSetException() = default;
    ~MappingNotSetException() noexcept final = default;
    const char* what() const noexcept final { return "Mapping file not set"; }
    void print(std::ostream& stream) const;
  };

  PixelMapper(MappingType_t mappingtype);
  ~PixelMapper() = default;

  ChipPosition getPosition(unsigned int feeID, unsigned int laneID, unsigned int chipID) const;
  ChipPosition getPosition(unsigned int feeID, const PixelChip& chip) const
  {
    return getPosition(feeID, chip.mLaneID, chip.mChipID);
  };

  int getNumberOfColumns() const { return mNumberOfColumns; }
  int getNumberOfRows() const { return mNumberOfRows; }
  MappingType_t getMappingType() const { return mMappingType; }

  void setMappingFile(const std::string_view mappingfile, MappingType_t mappingtype)
  {
    mMappingFile = mappingfile;
    mMappingType = mappingtype;
    init();
  }

 private:
  void checkInitialized() const;
  void init();
  std::unordered_map<ChipIdentifier, ChipPosition, ChipIdentifierHasher> mMapping;
  std::string mMappingFile;
  MappingType_t mMappingType = MappingType_t::MAPPING_UNKNOWN;
  int mNumberOfColumns = 0;
  int mNumberOfRows = 0;

  ClassDefNV(PixelMapper, 1);
};

std::ostream& operator<<(std::ostream& stream, const PixelMapper::InvalidChipException& error);
std::ostream& operator<<(std::ostream& stream, const PixelMapper::UninitException& error);
std::ostream& operator<<(std::ostream& stream, const PixelMapper::MappingNotSetException& error);

} // namespace o2::focal

#endif // ALICEO2_FOCAL_PixelMapper_H