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

/// \file T0FitHistos.cxx
/// \brief Class to store the TRD PH values for each chamber

#include "DataFormatsTRD/T0FitHistos.h"
#include <fairlogger/Logger.h>
#include <algorithm>

using namespace o2::trd;
using namespace o2::trd::constants;

void T0FitHistos::fill(const std::vector<o2::trd::PHData>& data)
{
  for (const auto& ph : data) {
    int det = ph.getDetector();
    int tb = ph.getTimebin();
    int adc = ph.getADC();

    if (ph.getNneighbours() != 2) {
      continue;
    }

    mDet.push_back(det);
    mTB.push_back(tb);
    mADC.push_back(adc);
    ++mNEntriesTot;
  }
}

void T0FitHistos::merge(const T0FitHistos* prev)
{
  auto sizePrev = (int)prev->getNEntries();

  for (int i = 0; i < sizePrev; ++i) {
    mDet.push_back(prev->getDetector(i));
    mTB.push_back(prev->getTimeBin(i));
    mADC.push_back(prev->getADC(i));
  }
}

void T0FitHistos::print()
{
  LOG(info) << "There are " << mNEntriesTot << " entries in the container";
}
