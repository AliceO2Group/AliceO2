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

#ifndef O2_MCH_DEVIO_DIGITS_DIGIT_IO_V2_H
#define O2_MCH_DEVIO_DIGITS_DIGIT_IO_V2_H

#include "DigitSamplerImpl.h"
#include "DigitSinkImpl.h"
#include <utility>
#include <vector>

/*
 * Digit file format 0 is storing Digits version D0
 * with no ROF (i.e. just a set of nDigits|DigitsD0|...)
 */

namespace o2::mch::io::impl
{
/**
 * Reader for digit file format 0
 */
class DigitSamplerV2 : public DigitSamplerImpl
{
 public:
  void count(std::istream& in, size_t& ntfs, size_t& nrofs, size_t& ndigits) override;
  bool read(std::istream& in,
            std::vector<Digit>& digits,
            std::vector<ROFRecord>& rofs) override;
  void rewind(std::istream& in);

 private:
  int mCurrentROF{0};
};

/**
 * Writer for digit file format 0
 */
struct DigitSinkV2 : public DigitSinkImpl {

  /** write rofs, digits at the current position in the output stream
   * @param digits vector of Digits, must not be empty
   * @param rofs vector of ROFRecord, might be empty
   * @returns true if writing was successull, false otherwise
   */
  bool write(std::ostream& out,
             gsl::span<const Digit> digits,
             gsl::span<const ROFRecord> rofs) override;

  bool mBinary;
};

} // namespace o2::mch::io::impl
#endif
