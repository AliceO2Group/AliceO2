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

#include "DigitReaderImpl.h"
#include <vector>
#include "DataFormatsMCH/ROFRecord.h"
#include <utility>
#include "DigitWriterImpl.h"

namespace o2::mch::io::impl
{
class DigitReaderV1 : public DigitReaderImpl
{
 public:
  void count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits) override;
  bool read(std::istream& in,
            std::vector<Digit>& digits,
            std::vector<ROFRecord>& rofs) override;
  void rewind(std::istream& in);
};

struct DigitWriterV1 : public DigitWriterImpl {
  /** write rofs, digits at the current position in the output stream
  * @param digits vector of Digits, must not be empty
  * @param rofs vector of ROFRecord, might be empty
  * @returns true if writing was successull, false otherwise
  */
  bool write(std::ostream& out,
             gsl::span<const Digit> digits,
             gsl::span<const ROFRecord> rofs) override;
};

} // namespace o2::mch::io::impl
