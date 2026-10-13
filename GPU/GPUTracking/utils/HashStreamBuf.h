// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HashStreamBuf.h
/// \author Felix Weiglhofer
/// \brief Stream buffer that supports on the fly hashing with an optional backing file buffer,
//         if we want to access the hash and file contents at the same time.

#include "Framework/SHA1.h"

#include <cstdint>
#include <streambuf>
#include <ostream>

class HashStreamBuf : public std::streambuf
{
 public:
  HashStreamBuf() = default;

  HashStreamBuf(bool doHash,
                std::streambuf* backing = nullptr)
    : mBacking(backing),
      mDoHash(doHash)
  {
    o2::framework::internal::SHA1Init(&mSHA1);
  }

  // Returns the hash without modifying the current hashing state.
  std::string hash() const
  {
    auto copy = mSHA1;
    unsigned char digest[20];
    o2::framework::internal::SHA1Final(digest, &copy);

    static constexpr char hex[] = "0123456789ABCDEF";
    std::string result;
    result.reserve(40);

    for (unsigned char byte : digest) {
      result += hex[byte >> 4];
      result += hex[byte & 0x0f];
    }

    return result;
  }

 protected:
  std::streamsize xsputn(const char* data,
                         std::streamsize size) override
  {
    if (size <= 0) {
      size = 0;
    }

    if (mBacking) {
      // Only hash bytes that were successfully written
      // to the backing stream.
      size = mBacking->sputn(data, size);
    }

    updateHash(data, size_t(size));
    return size;
  }

  int_type overflow(int_type ch) override
  {
    if (traits_type::eq_int_type(ch, traits_type::eof()))
      return traits_type::not_eof(ch);

    const char c = traits_type::to_char_type(ch);

    if (mBacking) {
      const auto result = mBacking->sputc(c);

      if (traits_type::eq_int_type(result,
                                   traits_type::eof())) {
        return traits_type::eof();
      }
    }

    updateHash(&c, 1);

    return ch;
  }

  int sync() override
  {
    int s = 0;
    if (mBacking) {
      s = mBacking->pubsync();
    }
    return s;
  }

 private:
  std::streambuf* mBacking = nullptr;
  o2::framework::internal::SHA1_CTX mSHA1;
  bool mDoHash = false;

  void updateHash(const char* data, size_t size)
  {
    if (mDoHash) {
      o2::framework::internal::SHA1Update(&mSHA1, reinterpret_cast<const unsigned char*>(data), size);
    }
  }
};
