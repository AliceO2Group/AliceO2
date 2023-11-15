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

#ifndef O2_MCH_DEVIO_DIGITS_DIGIT_SINK_IMPL_H
#define O2_MCH_DEVIO_DIGITS_DIGIT_SINK_IMPL_H

#include <ostream>
#include <gsl/span>
#include <memory>

namespace o2::mch
{
class Digit;
class ROFRecord;
} // namespace o2::mch

namespace o2::mch::io::impl
{

struct DigitSinkImpl {
  virtual ~DigitSinkImpl() = default;

  /** write rofs, digits at the current position in the output stream
   * @param digits vector of Digits, must not be empty
   * @param rofs vector of ROFRecord, might be empty
   * @returns true if writing was successull, false otherwise
   */
  virtual bool write(std::ostream& out,
                     gsl::span<const Digit> digits,
                     gsl::span<const ROFRecord> rofs) = 0;
};

std::unique_ptr<DigitSinkImpl> createDigitSinkImpl(int version);
} // namespace o2::mch::io::impl
#endif
