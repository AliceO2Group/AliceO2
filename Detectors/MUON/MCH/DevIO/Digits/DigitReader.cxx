// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DigitReader.h"
#include <istream>
#include "DigitFileFormat.h"
#include <iostream>
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <fmt/format.h>
#include "DigitReaderImpl.h"

namespace o2::mch::io
{

DigitReader::DigitReader(std::istream& in) : mInput{in}
{
  mFileFormat = readDigitFileFormat(mInput);
  mImpl = impl::createDigitReaderImpl(mFileFormat.fileVersion);
}

DigitReader::~DigitReader() = default;

bool DigitReader::read(std::vector<Digit>& digits,
                       std::vector<ROFRecord>& rofs)
{
  digits.clear();
  rofs.clear();
  bool ok = mImpl->read(mInput, digits, rofs);
  return ok;
}

void DigitReader::rewind()
{
  mImpl->rewind(mInput);
}

void DigitReader::count() const
{
  if (mCountDone) {
    return;
  }
  mImpl->count(mInput, mNofTimeFrames, mNofROFs, mNofDigits);
  mCountDone = true;
}

size_t DigitReader::nofTimeFrames() const
{
  count();
  return mNofTimeFrames;
}

size_t DigitReader::nofROFs() const
{
  count();
  return mNofROFs;
}

size_t DigitReader::nofDigits() const
{
  count();
  return mNofDigits;
}

} // namespace o2::mch::io
