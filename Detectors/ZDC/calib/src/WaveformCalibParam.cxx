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

#include "ZDCCalib/WaveformCalibParam.h"

using namespace o2::zdc;

void WaveformCalibParam::assign(const WaveformCalibData& data)
{
  for (int isig = 0; isig < NChannels; isig++) {
    float entries = data.getEntries(isig);
    int peak = data.mPeak;
    if (entries > 0) {
      int ifirst = data.getFirstValid(isig);
      int ilast = data.getLastValid(isig);
      channels[isig].ampMinID = peak - ifirst;
      for (int ip = ifirst; ip <= ilast; ip++) {
        channels[isig].shape.push_back(data.mWave[isig].mData[ip] / entries);
      }
    }
  }
}

void WaveformCalibChParam::print() const
{
  if (shape.size() > 0) {
    printf("Shape min at bin %d/%d\n", ampMinID, shape.size());
  } else {
    printf("No data\n", ampMinID, shape.size());
  }
}

void WaveformCalibParam::print() const
{
  for (int i = 0; i < NChannels; i++) {
    printf("%s ", channelName(i));
    channels[i].print();
  }
}
