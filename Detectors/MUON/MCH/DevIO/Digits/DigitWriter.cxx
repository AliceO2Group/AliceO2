// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include <set>
#include <fmt/format.h>
#include "DigitWriter.h"
#include <iostream>
#include "DigitWriterImpl.h"
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

  for (auto i = 0; i < rofs.size(); i++) {
    auto const& r = rofs[i];
    o2::InteractionRecord iri{r.getBCData().bc,
                              r.getBCData().orbit};
    minTimeDistances[r] = std::numeric_limits<int64_t>::max();
    for (auto j = 0; j < rofs.size(); j++) {
      if (i == j) {
        continue;
      }
      o2::InteractionRecord irj{rofs[j].getBCData().bc,
                                rofs[j].getBCData().orbit};
      auto d = irj.differenceInBC(iri);
      if (d >= 0) {
        minTimeDistances[rofs[i]] = std::min(minTimeDistances[rofs[i]], d);
      }
    }
  }
  return minTimeDistances;
}

void printRofs(std::ostream& os, gsl::span<const o2::mch::ROFRecord> rofs)
{
  auto minTimeDistances = computeMinTimeDistances(rofs);

  for (auto i = 0; i < rofs.size(); i++) {
    auto const& r = rofs[i];
    o2::InteractionRecord iri{r.getBCData().bc,
                              r.getBCData().orbit};
    minTimeDistances[r] = std::numeric_limits<int64_t>::max();
    for (auto j = 0; j < rofs.size(); j++) {
      if (i == j) {
        continue;
      }
      o2::InteractionRecord irj{rofs[j].getBCData().bc,
                                rofs[j].getBCData().orbit};
      auto d = irj.differenceInBC(iri);
      if (d >= 0) {
        minTimeDistances[rofs[i]] = std::min(minTimeDistances[rofs[i]], d);
      }
    }
  }

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
    printDigitsAndRofs(mOutput, digits, rofs);
  }
  return ok;
}

} // namespace o2::mch::io
