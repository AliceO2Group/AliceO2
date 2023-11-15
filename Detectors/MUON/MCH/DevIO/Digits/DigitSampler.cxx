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

#include "DigitSampler.h"
#include <istream>
#include "DigitFileFormat.h"
#include <iostream>
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <fmt/format.h>
#include "DigitSamplerImpl.h"

namespace o2::mch::io
{

DigitSampler::DigitSampler(std::istream& in) : mInput{in}
{
  mFileFormat = readDigitFileFormat(mInput);
  mImpl = impl::createDigitSamplerImpl(mFileFormat.fileVersion);
}

DigitSampler::~DigitSampler() = default;

bool DigitSampler::read(std::vector<Digit>& digits,
                        std::vector<ROFRecord>& rofs)
{
  digits.clear();
  rofs.clear();
  bool ok = mImpl->read(mInput, digits, rofs);
  return ok;
}

void DigitSampler::rewind()
{
  mImpl->rewind(mInput);
}

void DigitSampler::count() const
{
  if (mCountDone) {
    return;
  }
  mImpl->count(mInput, mNofTimeFrames, mNofROFs, mNofDigits);
  mCountDone = true;
}

size_t DigitSampler::nofTimeFrames() const
{
  count();
  return mNofTimeFrames;
}

size_t DigitSampler::nofROFs() const
{
  count();
  return mNofROFs;
}

size_t DigitSampler::nofDigits() const
{
  count();
  return mNofDigits;
}

} // namespace o2::mch::io
