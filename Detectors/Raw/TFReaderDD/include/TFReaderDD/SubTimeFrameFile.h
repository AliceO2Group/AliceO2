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

// Adapthed with minimal changes from Gvozden Nescovic code to read sTFs files created by DataDistribution

#ifndef _ALICEO2_SUBTIMEFRAME_FILE_RAWDD_H_
#define _ALICEO2_SUBTIMEFRAME_FILE_RAWDD_H_

#include <chrono>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

#include <Headers/DataHeader.h>
#include "Framework/Logger.h"

namespace o2
{
namespace rawdd
{

struct EquipmentIdentifier {
  o2::header::DataDescription mDataDescription;                   /* 2 x uint64_t */
  o2::header::DataHeader::SubSpecificationType mSubSpecification; /* uint32_t */
  o2::header::DataOrigin mDataOrigin;                             /* 1 x uint32_t */

  EquipmentIdentifier() = delete;

  EquipmentIdentifier(const o2::header::DataDescription& pDataDesc,
                      const o2::header::DataOrigin& pDataOrig,
                      const o2::header::DataHeader::SubSpecificationType& pSubSpec) noexcept
    : mDataDescription(pDataDesc),
      mSubSpecification(pSubSpec),
      mDataOrigin(pDataOrig)
  {
  }

  EquipmentIdentifier(const o2::header::DataIdentifier& pDataId,
                      const o2::header::DataHeader::SubSpecificationType& pSubSpec) noexcept
    : EquipmentIdentifier(pDataId.dataDescription, pDataId.dataOrigin, pSubSpec)
  {
  }

  EquipmentIdentifier(const EquipmentIdentifier& pEid) noexcept
    : EquipmentIdentifier(pEid.mDataDescription, pEid.mDataOrigin, pEid.mSubSpecification)
  {
  }

  EquipmentIdentifier(const o2::header::DataHeader& pDh) noexcept
    : EquipmentIdentifier(pDh.dataDescription, pDh.dataOrigin, pDh.subSpecification)
  {
  }

  operator o2::header::DataIdentifier() const noexcept
  {
    o2::header::DataIdentifier lRetId;
    lRetId.dataDescription = mDataDescription;
    lRetId.dataOrigin = mDataOrigin;
    return lRetId;
  }

  bool operator<(const EquipmentIdentifier& other) const noexcept
  {
    if (mDataDescription < other.mDataDescription) {
      return true;
    }

    if (mDataDescription == other.mDataDescription && mDataOrigin < other.mDataOrigin) {
      return true;
    }

    if (mDataDescription == other.mDataDescription && mDataOrigin == other.mDataOrigin &&
        mSubSpecification < other.mSubSpecification) {
      return true;
    }

    return false;
  }

  bool operator==(const EquipmentIdentifier& other) const noexcept
  {
    if (mDataDescription == other.mDataDescription &&
        mSubSpecification == other.mSubSpecification &&
        mDataOrigin == other.mDataOrigin) {
      return true;
    } else {
      return false;
    }
  }

  bool operator!=(const EquipmentIdentifier& other) const noexcept
  {
    return !(*this == other);
  }

  const std::string info() const
  {
    return fmt::format("{}/{}/{}",
                       std::string(mDataOrigin.str),
                       std::string(mDataDescription.str),
                       mSubSpecification);
  }
};

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileMeta
////////////////////////////////////////////////////////////////////////////////

struct SubTimeFrameFileMeta {
  static const o2::header::DataDescription sDataDescFileSubTimeFrame;

  static const o2::header::DataHeader getDataHeader()
  {
    auto lHdr = o2::header::DataHeader(
      SubTimeFrameFileMeta::sDataDescFileSubTimeFrame,
      o2::header::gDataOriginFLP,
      0, // TODO: subspecification? FLP ID? EPN ID?
      sizeof(SubTimeFrameFileMeta));

    lHdr.payloadSerializationMethod = o2::header::gSerializationMethodNone;

    return lHdr;
  }

  static constexpr std::uint64_t getSizeInFile()
  {
    return sizeof(o2::header::DataHeader) + sizeof(SubTimeFrameFileMeta);
  }

  ///
  /// Version of STF file format
  ///
  const std::uint64_t mStfFileVersion = 1;

  ///
  /// Size of the Stf in file, including this header.
  ///
  std::uint64_t mStfSizeInFile;

  ///
  /// Time when Stf was written (in ms)
  ///
  std::uint64_t mWriteTimeMs;

  auto getTimePoint()
  {
    using namespace std::chrono;
    return time_point<system_clock, milliseconds>{milliseconds{mWriteTimeMs}};
  }

  std::string getTimeString()
  {
    using namespace std::chrono;
    std::time_t lTime = system_clock::to_time_t(getTimePoint());

    std::stringstream lTimeStream;
    lTimeStream << std::put_time(std::localtime(&lTime), "%F %T");
    return lTimeStream.str();
  }

  SubTimeFrameFileMeta(const std::uint64_t pStfSize)
    : SubTimeFrameFileMeta()
  {
    mStfSizeInFile = pStfSize;
  }

  SubTimeFrameFileMeta()
    : mStfSizeInFile{0}
  {
    using namespace std::chrono;
    mWriteTimeMs = time_point_cast<milliseconds>(system_clock::now()).time_since_epoch().count();
  }

  friend std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileMeta& pMeta);
};

std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileMeta& pMeta);

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileDataIndex
////////////////////////////////////////////////////////////////////////////////

struct SubTimeFrameFileDataIndex {
  static const o2::header::DataDescription sDataDescFileStfDataIndex;

  struct DataIndexElem {
    /// Equipment Identifier: unrolled to pack better
    o2::header::DataDescription mDataDescription;
    o2::header::DataOrigin mDataOrigin;
    /// Number of data blocks <data_header, data>
    std::uint32_t mDataBlockCnt = 0;
    /// subspecification (u64)
    o2::header::DataHeader::SubSpecificationType mSubSpecification = 0;
    /// Offset of data block (corresponding data header) relative to
    std::uint64_t mOffset = 0;
    /// Total size of data blocks including headers
    std::uint64_t mSize = 0;

    DataIndexElem() = delete;
    DataIndexElem(const EquipmentIdentifier& pId,
                  const std::uint32_t pCnt,
                  const std::uint64_t pOff,
                  const std::uint64_t pSize)
      : mDataDescription(pId.mDataDescription),
        mDataOrigin(pId.mDataOrigin),
        mDataBlockCnt(pCnt),
        mSubSpecification(pId.mSubSpecification),
        mOffset(pOff),
        mSize(pSize)
    {
      static_assert(sizeof(DataIndexElem) == 48,
                    "DataIndexElem changed -> Binary compatibility is lost!");
    }
  };

  SubTimeFrameFileDataIndex() = default;

  void clear() noexcept { mDataIndex.clear(); }
  bool empty() const noexcept { return mDataIndex.empty(); }

  void AddStfElement(const EquipmentIdentifier& pEqDataId,
                     const std::uint32_t pCnt,
                     const std::uint64_t pOffset,
                     const std::uint64_t pSize)
  {
    mDataIndex.emplace_back(pEqDataId, pCnt, pOffset, pSize);
  }

  std::uint64_t getSizeInFile() const
  {
    return sizeof(o2::header::DataHeader) + (sizeof(DataIndexElem) * mDataIndex.size());
  }

  friend std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileDataIndex& pIndex);

 private:
  const o2::header::DataHeader getDataHeader() const
  {
    auto lHdr = o2::header::DataHeader(
      sDataDescFileStfDataIndex,
      o2::header::gDataOriginAny,
      0, // TODO: subspecification? FLP ID? EPN ID?
      mDataIndex.size() * sizeof(DataIndexElem));

    lHdr.payloadSerializationMethod = o2::header::gSerializationMethodNone;

    return lHdr;
  }

  std::vector<DataIndexElem> mDataIndex;
};

std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileDataIndex& pIndex);
} // namespace rawdd

} // namespace o2

namespace std
{
template <>
struct hash<o2::header::DataOrigin> {
  typedef o2::header::DataOrigin argument_type;
  typedef std::uint32_t result_type;

  result_type operator()(argument_type const& a) const noexcept
  {

    static_assert(sizeof(o2::header::DataOrigin::ItgType) == sizeof(uint32_t) &&
                    sizeof(o2::header::DataOrigin) == 4,
                  "DataOrigin must be 4B long (uint32_t itg[1])");
    return std::hash<o2::header::DataOrigin::ItgType>{}(a.itg[0]);
  }
};

template <>
struct hash<o2::header::DataDescription> {
  typedef o2::header::DataDescription argument_type;
  typedef std::uint64_t result_type;

  result_type operator()(argument_type const& a) const noexcept
  {

    static_assert(sizeof(o2::header::DataDescription::ItgType) == sizeof(uint64_t) &&
                    sizeof(o2::header::DataDescription) == 16,
                  "DataDescription must be 16B long (uint64_t itg[2])");

    return std::hash<o2::header::DataDescription::ItgType>{}(a.itg[0]) ^
           std::hash<o2::header::DataDescription::ItgType>{}(a.itg[1]);
  }
};

inline void hash_combine(std::size_t& seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

template <>
struct hash<o2::header::DataHeader> {
  typedef o2::header::DataHeader argument_type;
  typedef std::size_t result_type;

  result_type operator()(argument_type const& a) const noexcept
  {
    result_type h = (size_t(a.tfCounter) << 32) + a.subSpecification;
    hash_combine(h, a.dataOrigin, a.dataDescription);
    return h;
  }
};

template <>
struct hash<o2::header::DataIdentifier> {
  typedef o2::header::DataIdentifier argument_type;
  typedef std::uint64_t result_type;

  result_type operator()(argument_type const& a) const noexcept
  {

    return std::hash<o2::header::DataDescription>{}(a.dataDescription) ^
           std::hash<o2::header::DataOrigin>{}(a.dataOrigin);
  }
};

template <>
struct hash<o2::rawdd::EquipmentIdentifier> {
  typedef o2::rawdd::EquipmentIdentifier argument_type;
  typedef std::uint64_t result_type;

  result_type operator()(argument_type const& a) const noexcept
  {

    return std::hash<o2::header::DataDescription>{}(a.mDataDescription) ^
           (std::hash<o2::header::DataOrigin>{}(a.mDataOrigin) << 1) ^
           a.mSubSpecification;
  }
};

} //namespace std

#endif /* _ALICEO2_SUBTIMEFRAME_FILE_RAWDD_H_ */
