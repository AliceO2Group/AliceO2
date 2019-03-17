// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  static int GetPolynomialField(float nominalFieldkG, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field);

#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)

  /* Get pre-calculated polynomial field for the current ALICE field (if exists)
 */
  static int GetPolynomialField(GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field);

  /* Fit given field for TPC
 */
  static int FitFieldTpc(AliMagF* fld, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field, double step = 1.);

  /* Fit given field for TRD
 */
  static int FitFieldTrd(AliMagF* fld, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field, double step = 1.);

  /* Fit given field for ITS
 */
  static int FitFieldIts(AliMagF* fld, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field, double step = 1.);

#endif

  /* Get pre-calculated polynomial field of type "type", scaled with respect to nominalFieldkG
 */
  static int GetPolynomialField(StoredField_t type, float nominalFieldkG, GPUCA_NAMESPACE::gpu::GPUTPCGMPolynomialField& field);
};

#endif
