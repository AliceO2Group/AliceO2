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

#include <fstream>
#include <iostream>
#include <numeric>

#include "TROOT.h"

#include "TPCBase/CRUCalibHelpers.h"

using namespace o2::tpc;

/// return the hardware channel number as mapped in the CRU
int cru_calib_helpers::getHWChannel(int sampa, int channel, int regionIter)
{
  const int sampaOffet[5] = {0, 4, 8, 0, 4};
  if (regionIter && sampa == 2) {
    channel -= 16;
  }
  int outch = sampaOffet[sampa] + ((channel % 16) % 2) + 2 * (channel / 16) + (channel % 16) / 2 * 10;
  return outch;
}

/// convert HW mapping to sampa and channel number
std::tuple<int, int> cru_calib_helpers::getSampaInfo(int hwChannel, int cruID)
{
  static constexpr int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
  static constexpr int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};
  const int regionIter = cruID % 2;

  const int istreamm = ((hwChannel % 10) / 2);
  const int partitionStream = istreamm + regionIter * 5;
  const int sampaOnFEC = sampaMapping[partitionStream];
  const int channel = (hwChannel % 2) + 2 * (hwChannel / 10);
  const int channelOnSAMPA = channel + channelOffset[partitionStream];

  return {sampaOnFEC, channelOnSAMPA};
}

/// Test input channel mapping vs output channel mapping
///
/// Consistency check of mapping
void cru_calib_helpers::testChannelMapping(int cruID)
{
  const int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
  const int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};
  const int regionIter = cruID % 2;

  for (std::size_t ichannel = 0; ichannel < 80; ++ichannel) {
    const int istreamm = ((ichannel % 10) / 2);
    const int partitionStream = istreamm + regionIter * 5;
    const int sampaOnFEC = sampaMapping[partitionStream];
    const int channel = (ichannel % 2) + 2 * (ichannel / 10);
    const int channelOnSAMPA = channel + channelOffset[partitionStream];

    const size_t outch = cru_calib_helpers::getHWChannel(sampaOnFEC, channelOnSAMPA, regionIter);
    printf("%4zu %4d %4d : %4zu %s\n", outch, sampaOnFEC, channelOnSAMPA, ichannel, (outch != ichannel) ? "============" : "");
  }
}

/// debug differences between two cal pad objects
void cru_calib_helpers::debugDiff(std::string_view file1, std::string_view file2, std::string_view objName)
{
  using namespace o2::tpc;
  CalPad dummy;
  CalPad* calPad1{nullptr};
  CalPad* calPad2{nullptr};

  std::unique_ptr<TFile> tFile1(TFile::Open(file1.data()));
  std::unique_ptr<TFile> tFile2(TFile::Open(file2.data()));
  gROOT->cd();

  tFile1->GetObject(objName.data(), calPad1);
  tFile2->GetObject(objName.data(), calPad2);

  for (size_t iroc = 0; iroc < calPad1->getData().size(); ++iroc) {
    const auto& calArray1 = calPad1->getCalArray(iroc);
    const auto& calArray2 = calPad2->getCalArray(iroc);
    // skip empty
    if (!(std::abs(calArray1.getSum() + calArray2.getSum()) > 0)) {
      continue;
    }

    for (size_t ipad = 0; ipad < calArray1.getData().size(); ++ipad) {
      const auto val1 = calArray1.getValue(ipad);
      const auto val2 = calArray2.getValue(ipad);

      if (std::abs(val2 - val1) >= 0.25) {
        printf("%2zu %5zu : %.5f - %.5f = %.2f\n", iroc, ipad, val2, val1, val2 - val1);
      }
    }
  }
}
