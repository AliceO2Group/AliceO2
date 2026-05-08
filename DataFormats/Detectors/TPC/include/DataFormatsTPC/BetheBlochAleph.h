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

#ifndef AliceO2_TPC_BETHEBLOCH_H_
#define AliceO2_TPC_BETHEBLOCH_H_

#include "MathUtils/BetheBlochAleph.h"

namespace o2::tpc
{

template <typename T>
GPUdi() T BetheBlochAleph(T bg, T kp1, T kp2, T kp3, T kp4, T kp5)
{
  return o2::common::BetheBlochAleph(bg, kp1, kp2, kp3, kp4, kp5);
}

} // namespace o2::tpc

#endif
