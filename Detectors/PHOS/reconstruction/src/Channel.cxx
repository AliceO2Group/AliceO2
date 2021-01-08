// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "PHOSReconstruction/Channel.h"

using namespace o2::phos;

int Channel::getBranchIndex() const
{
  if (mHardwareAddress == -1) {
    throw HardwareAddressError();
  }
  return ((mHardwareAddress >> 11) & 0x1);
}

int Channel::getFECIndex() const
{
  if (mHardwareAddress == -1) {
    throw HardwareAddressError();
  }
  return ((mHardwareAddress >> 7) & 0xF);
}

Int_t Channel::getAltroIndex() const
{
  if (mHardwareAddress == -1) {
    throw HardwareAddressError();
  }
  return ((mHardwareAddress >> 4) & 0x7);
}

Int_t Channel::getChannelIndex() const
{
  if (mHardwareAddress == -1) {
    throw HardwareAddressError();
  }
  return (mHardwareAddress & 0xF);
}

Bunch& Channel::createBunch(uint8_t bunchlength, uint8_t starttime)
{
  mBunches.emplace_back(bunchlength, starttime);
  return mBunches.back();
}