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

#ifndef O2_MCH_DEVIO_DIGITS_DIGIT_READER_H
#define O2_MCH_DEVIO_DIGITS_DIGIT_READER_H

#include <iosfwd>
#include "DigitFileFormat.h"
#include <vector>
#include <memory>

namespace o2::mch
{
class Digit;
class ROFRecord;
} // namespace o2::mch

namespace o2::mch::io
{
namespace impl
{
class DigitSamplerImpl;
}
class DigitSampler
{
 public:
  DigitSampler(std::istream& in);

  /* defined in the implementation file, where mImpl is a complete type */
  ~DigitSampler();

  /** Which file format has been detected */
  DigitFileFormat fileFormat() const { return mFileFormat; }

  /** read rofs, digits at the current position in the input stream.
   * i.e. reads one full time frame.
   *
   * @param digits vector of Digits
   * @param rofs vector of ROFRecord
   * @returns true if reading was successull, false otherwise
   */
  bool read(std::vector<Digit>& digits,
            std::vector<ROFRecord>& rofs);

  /** Count the number of timeframes in the input stream.
   * WARNING : depending on the size of the input this might be a
   * costly operation */
  size_t nofTimeFrames() const;

  /** Count the number of ROFRecords in the input stream
   * WARNING : depending on the size of the input this might be a
   * costly operation */
  size_t nofROFs() const;

  /** Count the number of digits in the input stream
   * WARNING : depending on the size of the input this might be a
   * costly operation */
  size_t nofDigits() const;

  /** Rewind, aka restart reading from the beginning of the stream */
  void rewind();

 private:
  void count() const;

 private:
  std::istream& mInput;
  DigitFileFormat mFileFormat;
  std::unique_ptr<impl::DigitSamplerImpl> mImpl;
  mutable size_t mNofTimeFrames{0};
  mutable size_t mNofROFs{0};
  mutable size_t mNofDigits{0};
  mutable bool mCountDone{false};
};

} // namespace o2::mch::io

#endif
