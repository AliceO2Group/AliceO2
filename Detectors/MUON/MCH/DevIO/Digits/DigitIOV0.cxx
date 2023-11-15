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

#include "DigitIOV0.h"
#include "DigitD0.h"
#include <stdexcept>
#include <vector>
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <iostream>
#include <fmt/format.h>
#include "DigitFileFormat.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "IOStruct.h"

//
// V0 was prior to the introduction of rofs, so the file format is
// simply a set of consecutive [nofDigits|list of digits] blocks
//

namespace o2::mch::io::impl
{
void DigitSamplerV0::count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits)
{
  rewind(in);
  ndigits = 0;
  nrofs = 0;
  ntfs = 0;
  int ndig{0};
  auto dff = digitFileFormats[0];

  while ((ndig = advance(in, dff.digitSize, "digits")) >= 0) {
    ndigits += ndig;
    ++nrofs;
    ++ntfs;
  }
  rewind(in);
}

bool DigitSamplerV0::read(std::istream& in,
                          std::vector<Digit>& digits,
                          std::vector<ROFRecord>& rofs)
{
  // note the input vectors are not cleared as this is the responsability
  // of the calling class, if need be.
  std::vector<DigitD0> digitsd0;
  bool ok = readBinaryStruct(in, digitsd0, "digits");
  if (!ok) {
    return false;
  }

  for (auto d0 : digitsd0) {
    Digit d(d0.detID, d0.padID, d0.adc, d0.tfTime, d0.getNofSamples(), d0.isSaturated());
    digits.push_back(d);
  }

  rofs.emplace_back(o2::InteractionRecord(0, mCurrentROF), 0, digits.size());
  ++mCurrentROF;
  return true;
}

void DigitSamplerV0::rewind(std::istream& in)
{
  DigitSamplerImpl::rewind(in);
  mCurrentROF = 0;
}

bool DigitSinkV0::write(std::ostream& out,
                        gsl::span<const Digit> digits,
                        gsl::span<const ROFRecord> rofs)
{
  // V0 format had no notion of rofs, so we strip them
  for (auto r : rofs) {
    std::vector<DigitD0> digitsd0;
    for (int i = r.getFirstIdx(); i <= r.getLastIdx(); i++) {
      const Digit& d = digits[i];
      digitsd0.push_back(DigitD0{d.getTime(), d.getNofSamples(), d.getDetID(), d.getPadID(), d.getADC()});
      digitsd0.back().setSaturated(d.isSaturated());
    }
    gsl::span<const DigitD0> d0(digitsd0);
    bool ok = writeBinaryStruct(out, d0);
    if (!ok) {
      return false;
    }
  }
  return true;
}

} // namespace o2::mch::io::impl
