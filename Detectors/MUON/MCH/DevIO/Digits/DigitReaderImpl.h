// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma once

#include <istream>
#include "DigitFileFormat.h"
#include <fmt/format.h>
#include <memory>

namespace o2::mch
{
class Digit;
class ROFRecord;
} // namespace o2::mch

namespace o2::mch::io::impl
{

struct DigitReaderImpl {
  virtual ~DigitReaderImpl() = default;
  virtual void count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits) = 0;
  virtual bool read(std::istream& in,
                    std::vector<Digit>& digits,
                    std::vector<ROFRecord>& rofs) = 0;

  void rewind(std::istream& in);
};

std::unique_ptr<DigitReaderImpl> createDigitReaderImpl(int version);

} // namespace o2::mch::io::impl
