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

#include <TH1.h>
#include <TH2.h>
#include "ZDCBase/Constants.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include "DataFormatsZDC/RawEventData.h"
#ifndef ALICEO2_ZDC_INTERCALIB_H_
#define ALICEO2_ZDC_INTERCALIB_H_
namespace o2
{
namespace zdc
{
class InterCalib
{
 public:
  InterCalib() = default;
 int init();
 int run();

 private:
};
} // namespace zdc
} // namespace o2

#endif
