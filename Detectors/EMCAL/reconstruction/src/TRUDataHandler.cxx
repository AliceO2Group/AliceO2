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

#include <iostream>
#include "EMCALReconstruction/TRUDataHandler.h"

using namespace o2::emcal;

TRUDataHandler::TRUDataHandler()
{
  reset();
}

void TRUDataHandler::reset()
{
  mTRUIndex = -1;
  mL0Fired = false;
  mL0Time = -1;
  std::fill(mPatchTimes.begin(), mPatchTimes.end(), UCHAR_MAX);
}

void TRUDataHandler::printStream(std::ostream& stream) const
{
  std::string patchstring;
  for (auto index = 0; index < mPatchTimes.size(); index++) {
    if (hasPatch(index)) {
      if (patchstring.length()) {
        patchstring += ", ";
      }
      patchstring += std::to_string(index);
    }
  }
  if (!patchstring.length()) {
    patchstring = "-";
  }
  stream << "TRU: " << static_cast<int>(mTRUIndex) << ", time " << static_cast<int>(mL0Time) << ", fired: " << (mL0Fired ? "yes" : "no") << ", patches: " << patchstring;
}

TRUDataHandler::PatchIndexException::PatchIndexException(int8_t index) : mIndex(index), mMessage()
{
  mMessage = "Invalid patch index " + std::to_string(index);
}

void TRUDataHandler::PatchIndexException::printStream(std::ostream& stream) const
{
  stream << what();
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const TRUDataHandler& data)
{
  data.printStream(stream);
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const TRUDataHandler::PatchIndexException& error)
{
  error.printStream(stream);
  return stream;
}