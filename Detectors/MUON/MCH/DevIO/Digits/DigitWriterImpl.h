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

struct DigitWriterImpl {
  virtual ~DigitWriterImpl() = default;

  /** write rofs, digits at the current position in the output stream
  * @param digits vector of Digits, must not be empty
  * @param rofs vector of ROFRecord, might be empty
  * @returns true if writing was successull, false otherwise
  */
  virtual bool write(std::ostream& out,
                     gsl::span<const Digit> digits,
                     gsl::span<const ROFRecord> rofs) = 0;
};

template <typename T>
bool binary(std::ostream& os,
            gsl::span<const T> items)
{
  int nofItems = items.size();
  if (!nofItems) {
    return !os.bad();
  }
  os.write(reinterpret_cast<char*>(&nofItems), sizeof(int));
  os.write(reinterpret_cast<const char*>(items.data()), items.size_bytes());
  return !os.bad();
}

std::unique_ptr<DigitWriterImpl> createDigitWriterImpl(int version);
} // namespace o2::mch::io::impl
