// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include <set>
#include <fmt/format.h>
#include "DigitWriter.h"
#include <iostream>
#include "DigitWriterImpl.h"

namespace
{
template <typename T>
std::string asString(T t);

template <>
std::string asString(o2::mch::ROFRecord rof)
{
  return fmt::format("{} FirstIdx: {:5d} LastIdx: {:5d}",
                     rof.getBCData().asString(), rof.getFirstIdx(), rof.getLastIdx());
}
template <>
std::string asString(o2::mch::Digit d)
{
  return fmt::format("DetID {:4d} PadId {:10d} ADC {:10d} TFtime {:10d} NofSamples {:5d} {}",
                     d.getDetID(), d.getPadID(), d.getADC(), d.getTime(), d.nofSamples(),
                     d.isSaturated() ? "(S)" : "");
}
template <typename T>
void text(std::ostream& os,
          const char* headline,
          gsl::span<T> items)
{
  os << fmt::format("{:=^70}\n", fmt::format("{} {}", items.size(), headline));
  size_t i{0};
  for (const auto item : items) {
    os << fmt::format("[{:6d}] {}\n", i, asString(item));
    ++i;
  }
}

} // namespace

namespace o2::mch::io
{

DigitWriter::DigitWriter(std::ostream& os, DigitFileFormat format, size_t maxSize) : mOutput(os), mBinary(true), mFileFormat(format), mMaxSize(maxSize)
{
  // write the tag to identify the file
  os.write(reinterpret_cast<char*>(&mFileFormat), sizeof(DigitFileFormat));
  mImpl = impl::createDigitWriterImpl(mFileFormat.fileVersion);
}

DigitWriter::DigitWriter(std::ostream& os) : mOutput(os), mBinary(false), mImpl{}
{
}

DigitWriter::~DigitWriter() = default;

bool DigitWriter::write(gsl::span<const Digit> digits,
                        gsl::span<const ROFRecord> rofs)
{
  if (digits.empty()) {
    return false;
  }

  bool ok{true};

  if (mBinary) {
    auto pos = static_cast<size_t>(mOutput.tellp());
    auto newSize = (pos + digits.size_bytes() + rofs.size_bytes()) / 1024;
    if (newSize >= mMaxSize) {
      return false;
    }
    ok = mImpl->write(mOutput, digits, rofs);
  } else {
    text(mOutput, "rofs", rofs);
    text(mOutput, "digits", digits);
  }
  return ok;
}

} // namespace o2::mch::io
