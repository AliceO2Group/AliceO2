// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TOFCalibration/TOFFEElightConfig.h>

using namespace o2::tof;

const TOFFEEchannelConfig* TOFFEElightConfig::getChannelConfig(int icrate, int itrm, int ichain, int itdc, int ichtdc) const
{

  // return the channel config for the given crate, trm, chain, tdc, tdcchannel

  return icrate >= Geo::kNCrate ? nullptr : itrm >= Geo::kNTRM - 2 ? nullptr
                                          : ichain >= Geo::kNChain ? nullptr
                                          : itdc >= Geo::kNTdc     ? nullptr
                                          : ichtdc >= Geo::kNCh    ? nullptr
                                                                   : &mChannelConfig[icrate][itrm][ichain][itdc][ichtdc]; // the TRM index goes from 0 to 9, but in O2 kNTRM = 12, because it corresponds to the VME slots which span in [3, 12]
}
