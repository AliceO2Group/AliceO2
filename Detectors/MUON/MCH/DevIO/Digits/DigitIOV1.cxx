// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DigitIOV1.h"
#include "DigitD0.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DigitFileFormat.h"
#include "DigitReader.h"
#include <iostream>

namespace
{
std::pair<int, int> advanceOneEvent(std::istream& in)
{
  int nrofs = o2::mch::io::impl::advance(in, sizeof(o2::mch::ROFRecord), "rofs");
  if (nrofs < 0) {
    return std::make_pair(-1, -1);
  }
  int ndigits = o2::mch::io::impl::advance(in, sizeof(o2::mch::io::impl::DigitD0), "digits");
  return std::make_pair(nrofs, ndigits);
}
} // namespace

namespace o2::mch::io::impl
{
void DigitReaderV1::count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits)
{
  rewind(in);
  ndigits = 0;
  nrofs = 0;
  ntfs = 0;
  std::pair<int, int> pairs;
  std::pair<int, int> invalid{-1, -1};

  while ((pairs = advanceOneEvent(in)) != invalid) {
    ndigits += pairs.second;
    nrofs += pairs.first;
    ++ntfs;
  }
  rewind(in);
}

bool DigitReaderV1::read(std::istream& in,
                         std::vector<Digit>& digits,
                         std::vector<ROFRecord>& rofs)
{
  // note the input vectors are not cleared as this is the responsability
  // of the calling class, if need be.

  bool ok = readBinary(in, rofs, "rofs");
  if (!ok) {
    return false;
  }
  std::vector<DigitD0> digitsd0;

  ok = readBinary(in, digitsd0, "digits");
  if (!ok) {
    return false;
  }

  for (auto d0 : digitsd0) {
    Digit d(d0.detID, d0.padID, d0.adc, d0.tfTime, d0.getNofSamples(), d0.isSaturated());
    digits.push_back(d);
  }
  return true;
}

void DigitReaderV1::rewind(std::istream& in)
{
  DigitReaderImpl::rewind(in);
}

bool DigitWriterV1::write(std::ostream& out,
                          gsl::span<const Digit> digits,
                          gsl::span<const ROFRecord> rofs)
{
  if (rofs.empty()) {
    return false;
  }
  bool ok = impl::binary(out, rofs);

  std::vector<DigitD0> digitsd0;
  for (const auto& d : digits) {
    digitsd0.push_back(DigitD0{d.getTime(), d.nofSamples(), d.getDetID(), d.getPadID(), d.getADC()});
    digitsd0.back().setSaturated(d.isSaturated());
  }
  gsl::span<const DigitD0> d0(digitsd0);
  ok &= impl::binary(out, d0);
  return ok;
}

} // namespace o2::mch::io::impl
