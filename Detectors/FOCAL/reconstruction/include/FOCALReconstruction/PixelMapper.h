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

namespace o2::focal
{

class PixelMapping
{
 public:
  struct ChipIdentifier {
    unsigned int mLaneID;
    unsigned int mChipID;

    bool operator==(const ChipIdentifier& other) const { return mLaneID == other.mLaneID && mChipID == other.mChipID; }
  };
  struct ChipPosition {
    unsigned int mColumn;
    unsigned int mRow;
    bool mInvertColumn;
    bool mInvertRow;

    bool operator==(const ChipPosition& other) const { return mColumn == other.mColumn && mRow == other.mRow; }
  };
  struct ChipIdentifierHasher {

    /// \brief Functor implementation
    /// \param s ChipID for which to determine a hash value
    /// \return hash value for channel ID
    size_t operator()(const ChipIdentifier& s) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, s.mChipID);
      boost::hash_combine(seed, s.mLaneID);
      return seed;
    }
  };

  class InvalidChipException : public std::exception
  {
   public:
    InvalidChipException(unsigned int mappingVersion, PixelMapping::ChipIdentifier& identifier) : mMappingVersion(mappingVersion), mIdentifier(identifier), mMessage()
    {
      mMessage = "Invalid chip identifier for mapping " + std::to_string(mMappingVersion) + ": lane " + std::to_string(mIdentifier.mLaneID) + ", chip " + std::to_string(mIdentifier.mChipID);
    }
    ~InvalidChipException() noexcept final = default;

    const char* what() const noexcept final { return mMessage.data(); }
    const PixelMapping::ChipIdentifier& getIdentifier() const { return mIdentifier; }
    unsigned int getLane() const noexcept { return mIdentifier.mLaneID; }
    unsigned int getChipID() const noexcept { return mIdentifier.mChipID; }
    int getMappingVersion() const noexcept { return mMappingVersion; }
    void print(std::ostream& stream) const;

   private:
    unsigned int mMappingVersion;
    PixelMapping::ChipIdentifier mIdentifier;
    std::string mMessage;
  };

  class VersionException : public std::exception
  {
   public:
    VersionException(unsigned int version) : mMappingVersion(version), mMessage() {}
    ~VersionException() noexcept final = default;

    const char* what() const noexcept final { return mMessage.data(); }
    int getMappingVersion() const noexcept { return mMappingVersion; }
    void print(std::ostream& stream) const;

   private:
    unsigned int mMappingVersion;
    std::string mMessage;
  };

  PixelMapping() = default;
  PixelMapping(unsigned int version);
  virtual ~PixelMapping() = default;

  ChipPosition getPosition(unsigned int laneID, unsigned int chipID) const;
  ChipPosition getPosition(const PixelChip& chip) const
  {
    return getPosition(chip.mLaneID, chip.mChipID);
  };

  virtual unsigned int getNumberOfRows() const = 0;
  virtual unsigned int getNumberOfColumns() const = 0;

 protected:
  int mVersion = -1;
  bool mUseLanes = false;
  std::unordered_map<ChipIdentifier, ChipPosition, ChipIdentifierHasher> mMapping;
};

class PixelMappingOB : public PixelMapping
{
 public:
  PixelMappingOB() = default;
  PixelMappingOB(unsigned int version);
  ~PixelMappingOB() final = default;

  void init(unsigned int version);
  unsigned int getNumberOfRows() const final { return 6; }
  unsigned int getNumberOfColumns() const final { return 7; }

 private:
  void buildVersion0();
  void buildVersion1();
};

class PixelMappingIB : public PixelMapping
{
 public:
  PixelMappingIB() = default;
  PixelMappingIB(unsigned int version);
  ~PixelMappingIB() final = default;

  void init(unsigned int version);
  unsigned int getNumberOfRows() const final { return 6; }
  unsigned int getNumberOfColumns() const final { return 3; }

 private:
  void buildVersion0();
  void buildVersion1();
};

class PixelMapper
{
 public:
  enum class MappingType_t {
    MAPPING_IB,
    MAPPING_OB
  };
  PixelMapper(MappingType_t mappingtype);
  ~PixelMapper() = default;

  const PixelMapping& getMapping(unsigned int feeID) const;
  MappingType_t getMappingType() const { return mMappingType; }

 private:
  MappingType_t mMappingType;
  std::array<std::shared_ptr<PixelMapping>, 2> mMappings;
};

std::ostream& operator<<(std::ostream& stream, const PixelMapping::InvalidChipException& error);
std::ostream& operator<<(std::ostream& stream, const PixelMapping::VersionException& error);

} // namespace o2::focal

#endif // ALICEO2_FOCAL_PIXELMAPPER_H