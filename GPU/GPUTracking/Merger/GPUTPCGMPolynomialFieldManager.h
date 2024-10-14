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

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMPolynomialField;
}
} // namespace GPUCA_NAMESPACE

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

  GPUTPCGMPolynomialFieldManager() CON_DEFAULT;

  /* Get appropriate pre-calculated polynomial field for the given field value nominalFieldkG
 */
  static int32_t GetPolynomialField(float nominalFieldkG, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field);

#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)

  /* Get pre-calculated polynomial field for the current ALICE field (if exists)
 */
  static int32_t GetPolynomialField(GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field);

  /* Fit given field for TPC
 */
  static int32_t FitFieldTpc(AliMagF* fld, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field, double step = 1.);

  /* Fit given field for TRD
 */
  static int32_t FitFieldTrd(AliMagF* fld, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field, double step = 1.);

  /* Fit given field for ITS
 */
  static int32_t FitFieldIts(AliMagF* fld, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field, double step = 1.);

#endif

  /* Get pre-calculated polynomial field of type "type", scaled with respect to nominalFieldkG
 */
  static int32_t GetPolynomialField(StoredField_t type, float nominalFieldkG, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field);
};

#endif
