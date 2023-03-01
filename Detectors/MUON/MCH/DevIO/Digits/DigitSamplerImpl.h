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

#ifndef O2_MCH_DEVIO_DIGITS_DIGIT_READER_IMPL_H
#define O2_MCH_DEVIO_DIGITS_DIGIT_READER_IMPL_H

#include <istream>
#include "DigitFileFormat.h"
#include <fmt/format.h>
#include <memory>
#include <vector>

namespace o2::mch
{
class Digit;
class ROFRecord;
} // namespace o2::mch

namespace o2::mch::io::impl
{

struct DigitSamplerImpl {
  virtual ~DigitSamplerImpl() = default;
  virtual void count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits) = 0;
  virtual bool read(std::istream& in,
                    std::vector<Digit>& digits,
                    std::vector<ROFRecord>& rofs) = 0;

  void rewind(std::istream& in);
};

std::unique_ptr<DigitSamplerImpl> createDigitSamplerImpl(int version);

} // namespace o2::mch::io::impl

#endif
