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

// @brief Polimorphic class to access either local or grid files with fixed sed ot methods

#include "TFReaderDD/BinFileOp.h"
#include "Framework/Logger.h"
#include "CommonUtils/StringUtils.h"
#include <sys/mman.h>
#include <TFileCacheRead.h>

namespace o2::rawdd
{

//_____________________________________________________________________
BinFileOp* BinFileOp::open(const std::string& name)
{
  BinFileOp* ptr = o2::utils::Str::beginsWith(name, "alien://") ? static_cast<BinFileOp*>(new BinFileOpGrid(name)) : static_cast<BinFileOp*>(new BinFileOpLocal(name));
  return ptr->isGood() ? ptr : nullptr;
}

//_____________________________________________________________________
BinFileOpLocal::BinFileOpLocal(const std::string& name) : BinFileOp(name)
{
  mFileMap.open(name);
  if (!mFileMap.is_open()) {
    LOG(error) << "Failed to open TF file for reading (mmap).";
    return;
  }
  mFileSize = mFileMap.size();
  mFileOffset = 0;

#if __linux__
  madvise((void*)mFileMap.data(), mFileMap.size(), MADV_HUGEPAGE | MADV_SEQUENTIAL | MADV_DONTDUMP);
#endif
}

BinFileOpLocal::~BinFileOpLocal()
{
  if (!mFileMap.is_open()) {
#if __linux__
    madvise((void*)mFileMap.data(), mFileMap.size(), MADV_DONTNEED);
#endif
    mFileMap.close();
  }
}

bool BinFileOpLocal::read_advance(void* ptr, size_t len)
{
  if (!mFileMap.is_open()) {
    return false;
  }
  assert(mFileOffset <= mFileSize);
  const size_t lToRead = std::min(len, mFileSize - mFileOffset);
  if (lToRead != len) {
    LOGP(error, "BinFileOpLocal: request to read beyond the file end. pos={} size={} len={}, closing the file {}", mFileOffset, mFileSize, len, mFileName);
    mFileMap.close();
    mFileOffset = 0;
    mFileSize = 0;
    return false;
  }
  std::memcpy(reinterpret_cast<char*>(ptr), mFileMap.data() + mFileOffset, lToRead);
  mFileOffset += lToRead;
  return true;
}

unsigned char* BinFileOpLocal::bufferize(size_t& s)
{
  if (s > MaxBuffSize) {
    LOGP(fatal, "Requested buffer size {} exceeds max allowed {}", s, MaxBuffSize);
  }
  s = std::min(s, distance_to_eof());
  mBufferizedPos = position();
  return const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(mFileMap.data() + mFileOffset));
}

bool BinFileOpLocal::ignore_nbytes(size_t len)
{
  const size_t lToIgnore = std::min(len, std::size_t(mFileSize - mFileOffset));
  if (len != lToIgnore) {
    LOGP(error, "BinFileOpLocal: request to ignore bytes beyond the file end. pos={} size={} len={}, closing the file {}", mFileOffset, mFileSize, len, mFileName);
    mFileMap.close();
    mFileOffset = 0;
    mFileSize = 0;
    return false;
  }
  mFileOffset += lToIgnore;
  assert(mFileOffset <= mFileSize);
  return true;
}

//_____________________________________________________________________
BinFileOpGrid::BinFileOpGrid(const std::string& name) : BinFileOp(name)
{
  mFile.reset(TFile::Open(fmt::format("{}?filetype=raw", name).c_str()));
  if (!isGood()) {
    LOGP(error, "Failed to open file {} for reading.", name);
    return;
  }
  mFileSize = mFile->GetSize();
  set_position(0);
  mBuffer.reserve(MaxBuffSize);
}

void BinFileOpGrid::set_position(size_t pos)
{
  assert(pos <= mFileSize);
  mFileOffset = std::min(pos, mFileSize);
  mFile->Seek(mFileOffset);
}

bool BinFileOpGrid::read_advance(void* ptr, size_t len)
{
  if (!isGood()) {
    return false;
  }
  assert(mFileOffset <= mFileSize);
  const size_t lToRead = std::min(len, distance_to_eof());
  if (lToRead != len) {
    LOGP(error, "BinFileOpGrid: request to read beyond the file end. pos={} size={} len={}, closing the file {}", mFileOffset, mFileSize, len, mFileName);
    mFile.reset();
    mFileOffset = 0;
    mFileSize = 0;
    return false;
  }

  if (len < MaxBuffSize) {
    auto pos = position();
    LOGP(debug, "read_advance(fast) {} from {}", lToRead, position());
    void* lptr = bufferize(len);
    std::memcpy(ptr, lptr, len);
    set_position(pos + len);
  } else { // too large chunk to bufferize, read directly
    LOGP(debug, "read_advance(slow) {} from {}", lToRead, position());
    if (mFile->ReadBuffer(reinterpret_cast<char*>(ptr), lToRead)) {
      LOGP(error, "BinFileOpGrid: failed to read {} bytes from position {} of file {}, closing it", lToRead, mFileOffset, mFileName);
    }
    mFileOffset += lToRead;
  }
  return true;
}

unsigned char* BinFileOpGrid::bufferize(size_t& s)
{
  if (s > MaxBuffSize) {
    LOGP(fatal, "Requested buffer size {} exceeds max allowed {}", s, MaxBuffSize);
  }
  if (mBufferizedPos <= position() && mBufferizedPos + mBuffer.size() >= position() + s) {
    LOGP(debug, "bufferize(fast) {} from {}", s, position());
    return mBuffer.data() + (position() - mBufferizedPos);
  }
  s = std::min(distance_to_eof(), s);
  mBufferizedPos = position();
  mBuffer.resize(std::min(MaxBuffSize, distance_to_eof()));
  if (!mFile->ReadBuffer((char*)mBuffer.data(), mBuffer.size())) {
    LOGP(debug, "bufferize(slow) {} from {}", s, position());
    set_position(mBufferizedPos); // go back
    return mBuffer.data();
  }
  mBuffer.clear();
  LOGP(error, "BinFileOpGrid:bufferize failed to read {} bytes from position {} of file {}, closing it", s, mFileOffset, mFileName);
  set_position(mBufferizedPos);
  mBufferizedPos = -1UL;
  s = 0;
  return nullptr;
}

bool BinFileOpGrid::ignore_nbytes(size_t len)
{
  const size_t lToIgnore = std::min(len, distance_to_eof());
  if (len != lToIgnore) {
    LOGP(error, "BinFileOpGrid: request to ignore bytes beyond the file end. pos={} size={} len={}, closing the file {}", mFileOffset, mFileSize, len, mFileName);
    mFile.reset();
    mFileOffset = 0;
    mFileSize = 0;
    return false;
  }
  set_position(mFileOffset + lToIgnore);
  return true;
}

} // namespace o2::rawdd
