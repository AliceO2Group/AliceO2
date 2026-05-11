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

// @brief Polymorphic class to access either local or grid files with fixed sed ot methods
#ifndef _BIN_FILE_OP_H_
#define _BIN_FILE_OP_H_

#include <TFile.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <string>
#include <algorithm>

namespace o2::rawdd
{

class BinFileOp
{
 public:
  static constexpr size_t KBYTE = 1024;
  static constexpr size_t MBYTE = 1024 * KBYTE;
  static constexpr size_t MaxBuffSize = 20 * MBYTE;
  virtual ~BinFileOp() = default;

  virtual bool read_advance(void* ptr, size_t len) = 0;
  virtual void set_position(size_t pos) = 0;
  virtual bool ignore_nbytes(size_t pLen) = 0;
  virtual bool isGood() const = 0;
  virtual unsigned char* bufferize(size_t& s) = 0;
  virtual size_t bufferized_size() const = 0;
  size_t bufferized_pos() const { return mBufferizedPos; }
  size_t size() const { return mFileSize; }
  size_t position() const { return mFileOffset; }
  size_t distance_to_eof() const { return mFileSize - mFileOffset; }
  bool eof() const { return mFileOffset == mFileSize; }

  static BinFileOp* open(const std::string& name);

 protected:
  BinFileOp(const std::string& name) : mFileName(name) {}

  std::string mFileName = {};
  size_t mFileOffset = 0;
  size_t mFileSize = 0;
  size_t mBufferizedPos = -1UL;
};

//========================================================================
class BinFileOpLocal : public BinFileOp
{
 public:
  BinFileOpLocal(const std::string& name);
  ~BinFileOpLocal() override;

  bool read_advance(void* ptr, size_t len) override;
  void set_position(size_t pos) override
  {
    assert(pos <= mFileSize);
    mFileOffset = std::min(pos, mFileSize);
  }
  unsigned char* bufferize(size_t& s) override;
  size_t bufferized_size() const override { return mFileSize - mBufferizedPos; }

  bool ignore_nbytes(size_t len) override;
  bool isGood() const override { return mFileMap.is_open(); }

  size_t size() const { return mFileSize; }
  size_t position() const { return mFileOffset; }
  bool eof() const { return mFileOffset == mFileSize; }

 protected:
  boost::iostreams::mapped_file_source mFileMap;
};

//========================================================================
class BinFileOpGrid : public BinFileOp
{
 public:
  BinFileOpGrid(const std::string& name);
  ~BinFileOpGrid() override = default;

  bool read_advance(void* ptr, size_t len) override;
  unsigned char* bufferize(size_t& s) override;
  size_t bufferized_size() const override { return mBuffer.size(); }
  void set_position(size_t pos) override;
  bool ignore_nbytes(size_t len) override;
  bool isGood() const override { return mFile && !mFile->IsZombie(); }

  size_t size() const { return mFileSize; }
  size_t position() const { return mFileOffset; }
  bool eof() const { return mFileOffset == mFileSize; }

 protected:
  std::unique_ptr<TFile> mFile;
  std::vector<unsigned char> mBuffer;
};

} // namespace o2::rawdd

#endif
