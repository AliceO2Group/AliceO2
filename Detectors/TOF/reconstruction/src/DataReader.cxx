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

/// \file PixelReader.cxx
/// \brief Implementation of the ITS pixel reader class

#include "TOFReconstruction/DataReader.h"
#include "TOFBase/Geo.h"
#include <algorithm>
#include <fairlogger/Logger.h> // for LOG

using namespace o2::tof;
using o2::tof::Digit;

//______________________________________________________________________________
Bool_t DigitDataReader::getNextStripData(StripData& stripData)
{

  // getting the next strip that needs to be clusterized

  stripData.clear();
  if (!mLastDigit) {
    if (mIdx >= mDigitArray->size()) {
      return kFALSE;
    }
    mLastDigit = &((*mDigitArray)[mIdx++]);
  }

  stripData.stripID = mLastDigit->getChannel() / Geo::NPADS;

  stripData.digits.emplace_back(*mLastDigit);

  mLastDigit = nullptr;

  while (mIdx < mDigitArray->size()) {
    mLastDigit = &((*mDigitArray)[mIdx++]);
    if (stripData.stripID != mLastDigit->getChannel() / Geo::NPADS) {
      break;
    }
    stripData.digits.emplace_back(*mLastDigit);
    mLastDigit = nullptr;
  }

  // sorting the digits of the current strip according to the TDC
  std::sort(stripData.digits.begin(), stripData.digits.end(),
            [](const Digit& a, const Digit& b) { if(a.getBC() != b.getBC()){ return a.getBC() < b.getBC();} return a.getTDC() < b.getTDC(); });

  return kTRUE;
}

//______________________________________________________________________________
Bool_t RawDataReader::getNextStripData(DataReader::StripData& stripData) { return kTRUE; }
