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

#ifndef ALICEO2_SUBTIMEFRAME_FILE_READER_RAWDD_H_
#define ALICEO2_SUBTIMEFRAME_FILE_READER_RAWDD_H_

#include <Headers/DataHeader.h>
#include <Headers/Stack.h>
#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>
#include <Framework/OutputRoute.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <fstream>
#include <vector>

namespace o2f = o2::framework;

namespace o2
{

namespace rawdd
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileReader
////////////////////////////////////////////////////////////////////////////////
using MessagesPerRoute = std::unordered_map<std::string, std::unique_ptr<FairMQParts>>;

class SubTimeFrameFileReader
{
 public:
  static constexpr o2::header::DataDescription gDataDescSubTimeFrame{"DISTSUBTIMEFRAME"};
  struct STFHeader { // fake header to mimic DD SubTimeFrame::Header sent with DISTSUBTIMEFRAME message
    uint64_t id = uint64_t(-1);
    uint32_t firstOrbit = uint32_t(-1);
    std::uint32_t runNumber = 0;
  };

  SubTimeFrameFileReader() = delete;
  SubTimeFrameFileReader(const std::string& pFileName);
  ~SubTimeFrameFileReader();

  /// Read a single TF from the file
  std::unique_ptr<MessagesPerRoute> read(FairMQDevice* device, const std::vector<o2f::OutputRoute>& outputRoutes, const std::string& rawChannel, int verbosity);

  /// Tell the current position of the file
  inline std::uint64_t position() const { return mFileMapOffset; }

  /// Set the current position of the file
  inline void set_position(std::uint64_t pPos)
  {
    const std::uint64_t lPos = std::min(pPos, mFileSize);
    assert(pPos == lPos);
    mFileMapOffset = lPos;
  }

  /// Is the stream position at EOF
  inline bool eof() const { return mFileMapOffset == mFileSize; }

  /// Tell the size of the file
  inline std::uint64_t size() const { return mFileSize; }

 private:
  std::string mFileName;
  boost::iostreams::mapped_file_source mFileMap;
  std::uint64_t mFileMapOffset = 0;
  std::uint64_t mFileSize = 0;

  // helper to make sure written chunks are buffered, only allow pointers
  template <typename pointer,
            typename = std::enable_if_t<std::is_pointer<pointer>::value>>
  bool read_advance(pointer pPtr, std::uint64_t pLen)
  {
    if (!mFileMap.is_open()) {
      return false;
    }

    assert(mFileMapOffset <= mFileSize);
    const std::uint64_t lToRead = std::min(pLen, mFileSize - mFileMapOffset);

    if (lToRead != pLen) {
      LOGP(ERROR, "FileReader: request to read beyond the file end. pos={} size={} len={}",
           mFileMapOffset, mFileSize, pLen);
      LOGP(ERROR, "Closing the file {}. The read data is invalid.", mFileName);
      mFileMap.close();
      mFileMapOffset = 0;
      mFileSize = 0;
      return false;
    }

    std::memcpy(reinterpret_cast<char*>(pPtr), mFileMap.data() + mFileMapOffset, lToRead);
    mFileMapOffset += lToRead;
    return true;
  }

  // return the pointer
  unsigned char* peek() const
  {
    return const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(mFileMap.data() + mFileMapOffset));
  }

  inline bool ignore_nbytes(const std::size_t pLen)
  {
    const std::size_t lToIgnore = std::min(pLen, std::size_t(mFileSize - mFileMapOffset));
    if (pLen != lToIgnore) {
      LOGP(ERROR, "FileReader: request to ignore bytes beyond the file end. pos={} size={} len={}",
           mFileMapOffset, mFileSize, pLen);
      LOGP(ERROR, "Closing the file {}. The read data is invalid.", mFileName);
      mFileMap.close();
      mFileMapOffset = 0;
      mFileSize = 0;
      return false;
    }

    mFileMapOffset += lToIgnore;
    assert(mFileMapOffset <= mFileSize);
    return true;
  }

  std::size_t getHeaderStackSize();
  o2::header::Stack getHeaderStack(std::size_t& pOrigsize);

  // flags for upgrading DataHeader versions
  static std::uint64_t sStfId; // TODO: add id to files metadata
};
} // namespace rawdd
} // namespace o2

#endif /* ALICEO2_SUBTIMEFRAME_FILE_READER_RAWDD_H_ */
