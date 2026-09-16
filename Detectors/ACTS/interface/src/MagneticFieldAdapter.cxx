// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file MagneticFieldAdapter.cxx
/// \author Paolo Butti
///

#include "ACTSInterface/MagneticFieldAdapter.h"

#include <stdexcept>

#include <TGeoGlobalMagField.h>

#include "Acts/Definitions/Units.hpp"
#include "Field/MagneticField.h"
#include "Framework/Logger.h"

namespace o2::acts
{

namespace
{
/// O2 stores the field in kGauss, ACTS in its native unit system.
constexpr double kGaussToActs = 0.1 * Acts::UnitConstants::T;
} // namespace

MagneticFieldAdapter::MagneticFieldAdapter()
  : MagneticFieldAdapter(static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField()))
{
}

MagneticFieldAdapter::MagneticFieldAdapter(o2::field::MagneticField* field) : mField(field)
{
  if (mField == nullptr) {
    throw std::runtime_error(
      "o2::acts::MagneticFieldAdapter: no O2 magnetic field available. Request GRPMagField in the "
      "GRPGeomRequest of the workflow, or initialise the field explicitly with "
      "o2::base::Propagator::initFieldFromGRP().");
  }
  LOG(info) << "ACTS magnetic field adapter bound to O2 field '" << mField->GetName() << "'";
}

Acts::MagneticFieldProvider::Cache MagneticFieldAdapter::makeCache(const Acts::MagneticFieldContext& /*mctx*/) const
{
  // The O2 field parametrisation keeps its own internal caches and offers no
  // per-client cache handle, so there is nothing to carry here.
  return Acts::MagneticFieldProvider::Cache(std::in_place_type<CacheImpl>);
}

Acts::Result<Acts::Vector3> MagneticFieldAdapter::getField(const Acts::Vector3& position,
                                                           Cache& /*cache*/) const
{
  const double xyz[3] = {position[0] / Acts::UnitConstants::cm,
                         position[1] / Acts::UnitConstants::cm,
                         position[2] / Acts::UnitConstants::cm};
  double bxyz[3] = {0., 0., 0.};
  mField->Field(xyz, bxyz);

  return Acts::Result<Acts::Vector3>::success(
    Acts::Vector3{bxyz[0] * kGaussToActs, bxyz[1] * kGaussToActs, bxyz[2] * kGaussToActs});
}

} // namespace o2::acts
