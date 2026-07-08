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

/// \file GPUTPCGMPolynomialFieldManager.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMPOLYNOMIALFIELDMANAGER_H
#define GPUTPCGMPOLYNOMIALFIELDMANAGER_H

#include "GPUCommonDef.h"
class AliMagF;

namespace o2::gpu
{
class GPUTPCGMPolynomialField;
} // namespace o2::gpu

/**
 * @class GPUTPCGMPolynomialFieldManager
 *
 */

class GPUTPCGMPolynomialFieldManager
{
 public:
  enum StoredField_t { kUnknown,
                       kUniform,
                       k2kG,
                       k5kG }; // known fitted polynomial fields, stored in constants

  GPUTPCGMPolynomialFieldManager() = default;

  /* Get appropriate pre-calculated polynomial field for the given field value nominalFieldkG
   */
  static int32_t GetPolynomialField(float nominalFieldkG, o2::gpu::GPUTPCGMPolynomialField& field, float biasX=0.f, float biasY=0.f, float biasZ=0.f);

  /* Get pre-calculated polynomial field of type "type", scaled with respect to nominalFieldkG
   */
  static int32_t GetPolynomialField(StoredField_t type, float nominalFieldkG, o2::gpu::GPUTPCGMPolynomialField& field, float biasX=0.f, float biasY=0.f, float biasZ=0.f);
};

#endif
