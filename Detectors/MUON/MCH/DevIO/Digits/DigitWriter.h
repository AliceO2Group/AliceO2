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

#include <iosfwd>
#include <gsl/span>
#include "DigitFileFormat.h"
#include <memory>
#include <numeric>

namespace o2::mch
{
class ROFRecord;
class Digit;

namespace io
{
namespace impl
{
class DigitWriterImpl;
}
class DigitWriter
{
 public:
  /** Create a text digit writer
  * @param os output stream to write to 
  */
  DigitWriter(std::ostream& os);

  /** Create a binary digit writer
  * @param os output stream to write to 
  * @param dff the digit file format to be used
  * @param maxSize if not zero indicate that writing should stop past 
  * this size, expressed in KB.
  */
  DigitWriter(std::ostream& os, DigitFileFormat format,
              size_t maxSize = std::numeric_limits<size_t>::max());

  /* defined in the implementation file, where mImpl is a complete type */
  ~DigitWriter();

  /** write rofs, digits at the current position in the output stream
  * @param digits vector of Digits, must not be empty
  * @param rofs vector of ROFRecord, might be empty
  * @returns true if writing was successull, false otherwise
  */
  bool write(gsl::span<const Digit> digits,
             gsl::span<const ROFRecord> rofs = {});

 private:
  std::ostream& mOutput;                        // underlying stream used for output
  bool mBinary;                                 // whether to output in binary mode or text mode
  DigitFileFormat mFileFormat{};                // version information for the binary case
  std::unique_ptr<impl::DigitWriterImpl> mImpl; // actual implementation of the writer
  size_t mMaxSize;                              // max size written to output
};
} // namespace io
} // namespace o2::mch
