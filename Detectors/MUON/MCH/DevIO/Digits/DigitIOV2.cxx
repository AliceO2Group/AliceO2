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

#include "DigitIOV2.h"
#include <stdexcept>
#include <vector>
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <iostream>
#include <fmt/format.h>
#include "DigitFileFormat.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "IO.h"

//
// V2 was prior to the introduction of rofs, so the file format is
// simply a set of consecutive [nofDigits|list of digits] blocks
//

namespace o2::mch::io::impl
{
void DigitSamplerV2::count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits)
{
  auto dff = digitFileFormats[2];
  rewind(in);
  ndigits = 0;
  nrofs = 0;
  ntfs = 0;
  int ndig{0};

  while ((ndig = advance(in, dff.digitSize, "digits")) >= 0) {
    ndigits += ndig;
    ++nrofs;
    ++ntfs;
  }
  rewind(in);
}

bool DigitSamplerV2::read(std::istream& in,
                          std::vector<Digit>& digits,
                          std::vector<ROFRecord>& rofs)
{
  if (in.peek() == EOF) {
    return false;
  }
  // note the input vectors are not cleared as this is the responsability
  // of the calling class, if need be.
  int ndigits = readNofItems(in, "digits");

  for (int i = 0; i < ndigits; i++) {
    uint32_t tfTime;
    uint16_t nofSamples;
    uint8_t sat;
    uint32_t deID;
    uint32_t padID;
    uint32_t adc;
    in.read(reinterpret_cast<char*>(&tfTime), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&nofSamples), sizeof(uint16_t));
    in.read(reinterpret_cast<char*>(&sat), sizeof(uint8_t));
    in.read(reinterpret_cast<char*>(&deID), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&padID), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&adc), sizeof(uint32_t));
    digits.emplace_back(deID, padID, adc, tfTime, nofSamples, sat > 0);
  }

  rofs.emplace_back(o2::InteractionRecord(0, mCurrentROF), 0, ndigits);
  ++mCurrentROF;
  return !in.fail();
}

void DigitSamplerV2::rewind(std::istream& in)
{
  DigitSamplerImpl::rewind(in);
  mCurrentROF = 0;
}

bool DigitSinkV2::write(std::ostream& out,
                        gsl::span<const Digit> digits,
                        gsl::span<const ROFRecord> rofs)
{
  // V2 format had no notion of rofs, so we strip them
  for (auto r : rofs) {
    writeNofItems(out, r.getNEntries());
    for (int i = r.getFirstIdx(); i <= r.getLastIdx(); i++) {
      const Digit& d = digits[i];
      uint32_t tfTime = d.getTime();
      uint16_t nofSamples = d.getNofSamples();
      uint32_t deID = d.getDetID();
      uint32_t padID = d.getPadID();
      uint32_t adc = d.getADC();
      uint8_t sat = d.isSaturated();
      out.write(reinterpret_cast<const char*>(&tfTime), sizeof(uint32_t));
      out.write(reinterpret_cast<const char*>(&nofSamples), sizeof(uint16_t));
      out.write(reinterpret_cast<const char*>(&sat), sizeof(uint8_t));
      out.write(reinterpret_cast<const char*>(&deID), sizeof(uint32_t));
      out.write(reinterpret_cast<const char*>(&padID), sizeof(uint32_t));
      out.write(reinterpret_cast<const char*>(&adc), sizeof(uint32_t));
    }
  }
  return !out.fail();
}

} // namespace o2::mch::io::impl
