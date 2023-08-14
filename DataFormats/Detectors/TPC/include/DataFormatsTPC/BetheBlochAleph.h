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

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

namespace o2
{
namespace tpc
{

template <typename T>
GPUdi() T BetheBlochAleph(T bg, T kp1, T kp2, T kp3, T kp4, T kp5)
{
  T beta = bg / o2::gpu::GPUCommonMath::Sqrt(static_cast<T>(1.) + bg * bg);

  T aa = o2::gpu::GPUCommonMath::Pow(beta, kp4);
  T bb = o2::gpu::GPUCommonMath::Pow(static_cast<T>(1.) / bg, kp5);
  bb = o2::gpu::GPUCommonMath::Log(kp3 + bb);

  return (kp2 - aa - bb) * kp1 / aa;
}

} // namespace tpc
} // namespace o2

#endif
