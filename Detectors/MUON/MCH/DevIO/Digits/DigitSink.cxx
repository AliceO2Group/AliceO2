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

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include <set>
#include <fmt/format.h>
#include "DigitSink.h"
#include <iostream>
#include "DigitSinkImpl.h"
#include <map>
#include <algorithm>
#include <limits>

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
                     d.getDetID(), d.getPadID(), d.getADC(), d.getTime(), d.getNofSamples(),
                     d.isSaturated() ? "(S)" : "");
}

std::map<o2::mch::ROFRecord, int64_t> computeMinTimeDistances(gsl::span<const o2::mch::ROFRecord> rofs)
{
  std::map<o2::mch::ROFRecord, int64_t> minTimeDistances;

  for (auto i = 0; i < rofs.size() - 1; i++) {
    auto const& r = rofs[i];
    o2::InteractionRecord iri{r.getBCData().bc,
                              r.getBCData().orbit};
    minTimeDistances[r] = std::numeric_limits<int64_t>::max();
    auto j = i + 1;
    o2::InteractionRecord irj{rofs[j].getBCData().bc,
                              rofs[j].getBCData().orbit};
    auto d = irj.differenceInBC(iri);
    if (d >= 0) {
      minTimeDistances[rofs[i]] = std::min(minTimeDistances[rofs[i]], d);
    }
  }
  return minTimeDistances;
}

void printRofs(std::ostream& os, gsl::span<const o2::mch::ROFRecord> rofs)
{
  auto minTimeDistances = computeMinTimeDistances(rofs);

  os << fmt::format("{:=^70}\n", fmt::format("{} rofs", rofs.size()));
  size_t i{0};
  for (const auto& r : rofs) {
    os << fmt::format("[{:6d}] {}", i, asString(r));
    if (minTimeDistances[r] < 4) {
      os << fmt::format(" min distance {} !", minTimeDistances[r]);
    }
    os << "\n";
    ++i;
  }
}

struct DigitIdComparator {
  bool operator()(const o2::mch::Digit& d1, const o2::mch::Digit& d2) const
  {
    if (d1.getDetID() == d2.getDetID()) {
      return d1.getPadID() < d2.getPadID();
    }
    return d1.getDetID() < d2.getDetID();
  }
};

void printDigitsAndRofs(std::ostream& os,
                        gsl::span<const o2::mch::Digit> digits,
                        gsl::span<const o2::mch::ROFRecord> rofs)
{
  printRofs(os, rofs);
  os << fmt::format("{:=^70}\n", fmt::format("{} digits", digits.size()));
  size_t irof{0};
  size_t digitIndex{0};
  for (const auto& r : rofs) {
    os << fmt::format("{:-^95}\n", fmt::format("ROF {:4d} with {:5d} digits",
                                               irof, r.getNEntries()));
    ++irof;
    std::map<o2::mch::Digit, uint16_t, DigitIdComparator> dids;
    for (auto j = r.getFirstIdx(); j <= r.getLastIdx(); j++) {
      const auto& d = digits[j];
      dids[d]++;
    }
    size_t i{0};
    for (auto& p : dids) {
      const auto& d = p.first;
      os << fmt::format("[{:6d}] ({:6d}) {}\n", digitIndex, i, asString(d));
      ++digitIndex;
      ++i;
    }
    // check that, within a rof, each digit appears only once.
    // if not, report that as an error
    for (auto& p : dids) {
      if (p.second != 1) {
        os << "!!! ERROR : got a duplicated digit (not merged?) : " << p.first << " appears " << p.second << " times\n";
      }
    }
  }
}

} // namespace

namespace o2::mch::io
{

DigitSink::DigitSink(std::ostream& os, DigitFileFormat format, size_t maxSize) : mOutput(os), mBinary(true), mFileFormat(format), mMaxSize(maxSize)
{
  // write the tag to identify the file
  os.write(reinterpret_cast<char*>(&mFileFormat), sizeof(DigitFileFormat));
  mImpl = impl::createDigitSinkImpl(mFileFormat.fileVersion);
}

DigitSink::DigitSink(std::ostream& os) : mOutput(os), mBinary(false), mImpl{}
{
}

DigitSink::~DigitSink() = default;

bool DigitSink::write(gsl::span<const Digit> digits,
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
    printDigitsAndRofs(mOutput, digits, rofs);
  }
  return ok;
}

} // namespace o2::mch::io
