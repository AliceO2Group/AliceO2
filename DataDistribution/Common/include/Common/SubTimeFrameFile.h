// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_FILE_H_
#define ALICEO2_SUBTIMEFRAME_FILE_H_

#include <chrono>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <vector>

#include <Headers/DataHeader.h>

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileMeta
////////////////////////////////////////////////////////////////////////////////

struct SubTimeFrameFileMeta {
  static const o2::header::DataDescription sDataDescFileSubTimeFrame;

  static const o2::header::DataHeader getDataHeader()
  {
    return o2::header::DataHeader(
      SubTimeFrameFileMeta::sDataDescFileSubTimeFrame,
      o2::header::gDataOriginAny,
      0, // TODO: subspecification? FLP ID? EPN ID?
      sizeof(SubTimeFrameFileMeta));
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
    return time_point<system_clock, milliseconds>{ milliseconds{ mWriteTimeMs } };
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
    : mStfSizeInFile{ 0 }
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
    o2::header::DataIdentifier mDataIdentifier;
    std::uint64_t mOffset = 0;
    std::uint64_t mSize = 0;

    DataIndexElem() = delete;
    DataIndexElem(const o2::header::DataIdentifier& pId, const std::uint64_t pOff, const std::uint64_t pSize)
      : mDataIdentifier(pId),
        mOffset(pOff),
        mSize(pSize)
    {
      static_assert(sizeof(DataIndexElem) == 40, "DataIdentifier changed -> Binary compatibility is lost!");
    }
  };

  SubTimeFrameFileDataIndex() = default;

  void clear() noexcept { mDataIndex.clear(); }
  bool empty() const noexcept { return mDataIndex.empty(); }

  void AddStfElement(const o2::header::DataIdentifier& pDataId, const std::uint64_t pOffset, const std::uint64_t pSize)
  {
    mDataIndex.emplace_back(DataIndexElem(pDataId, pOffset, pSize));
  }

  const std::uint64_t getSizeInFile() const
  {
    return sizeof(o2::header::DataHeader) + sizeof(DataIndexElem) * mDataIndex.size();
  }

  friend std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileDataIndex& pIndex);

 private:
  const o2::header::DataHeader getDataHeader() const
  {
    return o2::header::DataHeader(
      sDataDescFileStfDataIndex,
      o2::header::gDataOriginAny,
      0, // TODO: subspecification? FLP ID? EPN ID?
      mDataIndex.size() * sizeof(DataIndexElem));
  }

  std::vector<DataIndexElem> mDataIndex;
};

std::ostream& operator<<(std::ostream& pStream, const SubTimeFrameFileDataIndex& pIndex);
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_FILE_H_ */
