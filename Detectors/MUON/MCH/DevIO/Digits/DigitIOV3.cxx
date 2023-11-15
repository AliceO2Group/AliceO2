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

#include "DigitIOV3.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DigitFileFormat.h"
#include "DigitSampler.h"
#include "IO.h"
#include "IOStruct.h"
#include <iostream>

namespace o2::mch::io::impl
{

void DigitSamplerV3::count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits)
{
  rewind(in);
  ndigits = 0;
  nrofs = 0;
  ntfs = 0;
  std::pair<int, int> pairs;
  std::pair<int, int> invalid{-1, -1};

  while ((pairs = advanceOneEvent(in, 3)) != invalid) {
    ndigits += pairs.second;
    nrofs += pairs.first;
    ++ntfs;
  }
  rewind(in);
}

bool DigitSamplerV3::read(std::istream& in,
                          std::vector<Digit>& digits,
                          std::vector<ROFRecord>& rofs)
{
  if (in.peek() == EOF) {
    return false;
  }
  // note the input vectors are not cleared as this is the responsability
  // of the calling class, if need be.

  int nrofs = readNofItems(in, "rofs");
  if (in.fail()) {
    return false;
  }

  for (auto i = 0; i < nrofs; i++) {
    uint16_t bc;
    uint32_t orbit;
    uint32_t firstIdx;
    uint32_t nentries;
    in.read(reinterpret_cast<char*>(&bc), sizeof(uint16_t));
    in.read(reinterpret_cast<char*>(&orbit), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&firstIdx), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&nentries), sizeof(uint32_t));
    rofs.emplace_back(o2::InteractionRecord{bc, orbit}, firstIdx, nentries);
    if (in.fail()) {
      return false;
    }
  }

  int ndigits = readNofItems(in, "digits");
  if (in.fail()) {
    return false;
  }

  for (int i = 0; i < ndigits; i++) {
    uint32_t tfTime;
    uint16_t nofSamples;
    uint32_t deID;
    uint32_t padID;
    uint32_t adc;
    uint8_t sat;
    in.read(reinterpret_cast<char*>(&tfTime), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&nofSamples), sizeof(uint16_t));
    in.read(reinterpret_cast<char*>(&sat), sizeof(uint8_t));
    in.read(reinterpret_cast<char*>(&deID), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&padID), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&adc), sizeof(uint32_t));
    digits.emplace_back(deID, padID, adc, tfTime, nofSamples, sat > 0);
    if (in.fail()) {
      return false;
    }
  }
  return true;
}

void DigitSamplerV3::rewind(std::istream& in)
{
  DigitSamplerImpl::rewind(in);
}

bool DigitSinkV3::write(std::ostream& out,
                        gsl::span<const Digit> digits,
                        gsl::span<const ROFRecord> rofs)
{
  if (rofs.empty()) {
    return false;
  }
  writeNofItems(out, rofs.size());
  if (out.fail()) {
    return false;
  }
  for (auto r : rofs) {
    uint16_t bc = r.getBCData().bc;
    uint32_t orbit = r.getBCData().orbit;
    uint32_t firstIdx = r.getFirstIdx();
    uint32_t nentries = r.getNEntries();
    out.write(reinterpret_cast<const char*>(&bc), sizeof(uint16_t));
    out.write(reinterpret_cast<const char*>(&orbit), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&firstIdx), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&nentries), sizeof(uint32_t));
    if (out.fail()) {
      return false;
    }
  }

  writeNofItems(out, digits.size());
  if (out.fail()) {
    return false;
  }
  for (const auto& d : digits) {
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
    if (out.fail()) {
      return false;
    }
  }
  return true;
}

} // namespace o2::mch::io::impl
