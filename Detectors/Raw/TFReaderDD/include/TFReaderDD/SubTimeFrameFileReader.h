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

#include "TFReaderDD/SubTimeFrameFile.h"
#include <Headers/DataHeader.h>
#include <Headers/STFHeader.h>
#include "DetectorsCommonDataFormats/DetID.h"
#include <Headers/Stack.h>
#include <fairmq/Parts.h>
#include <fairmq/Device.h>
#include <Framework/OutputRoute.h>
#include "TFReaderDD/BinFileOp.h"
// RSREM
// #include <boost/filesystem.hpp>
// #include <boost/iostreams/device/mapped_file.hpp>
#include <fstream>
#include <vector>
#include <unordered_map>

namespace o2f = o2::framework;

namespace o2
{

namespace rawdd
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileReader
////////////////////////////////////////////////////////////////////////////////
using MessagesPerRoute = std::unordered_map<std::string, std::unique_ptr<fair::mq::Parts>>;

class SubTimeFrameFileReader
{
 public:

  SubTimeFrameFileReader() = delete;
  SubTimeFrameFileReader(const std::string& pFileName, o2::detectors::DetID::mask_t detMask, int verb, bool sup0xccdb, bool repaireHeaders, bool rejectDistSTF);
  ~SubTimeFrameFileReader() = default;

  /// Read a single TF from the file
  std::unique_ptr<MessagesPerRoute> read(fair::mq::Device* device, const std::vector<o2f::OutputRoute>& outputRoutes, const std::string& rawChannel, size_t slice);

 private:
  std::unordered_map<o2::header::DataOrigin, bool> mDetOrigMap;

  std::unique_ptr<BinFileOp> mFile;
  // RSREM
  // std::string mFileName;
  // boost::iostreams::mapped_file_source mFileMap;
  // std::uint64_t mFileMapOffset = 0;
  // std::uint64_t mFileSize = 0;

  int mVerbosity = 0;
  bool mSup0xccdb = true;
  bool mRepaireHeaders = true;
  bool mRejectDistSTF = true;

  const std::string describeHeader(const o2::header::DataHeader& hd, bool full = false) const;

  // helper to make sure written chunks are buffered, only allow pointers
  template <typename pointer, typename = std::enable_if_t<std::is_pointer<pointer>::value>>
  inline bool read_advance(pointer pPtr, std::uint64_t pLen)
  {
    if (!mFile) {
      return false;
    }
    return mFile->read_advance(pPtr, pLen);
  }

  // return the pointer
  //  inline unsigned char* peek() { mFile->peek(); }

  //  inline bool ignore_nbytes(const std::size_t pLen) { mFle->ignore_nbytes(pLen); }

  std::size_t getHeaderStackSize();
  o2::header::Stack getHeaderStack(std::size_t& pOrigsize);

  // flags for upgrading DataHeader versions
  static std::uint64_t sStfId; // TODO: add id to files metadata
};
} // namespace rawdd
} // namespace o2

#endif /* ALICEO2_SUBTIMEFRAME_FILE_READER_RAWDD_H_ */
