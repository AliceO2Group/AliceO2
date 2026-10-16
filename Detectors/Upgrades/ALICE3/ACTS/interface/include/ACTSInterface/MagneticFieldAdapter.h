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
/// \file MagneticFieldAdapter.h
/// \brief Acts::MagneticFieldProvider backed by the O2 field
/// \author Paolo Butti
///

#ifndef ALICEO2_ACTS_MAGNETICFIELDADAPTER_H
#define ALICEO2_ACTS_MAGNETICFIELDADAPTER_H

#include "Acts/MagneticField/MagneticFieldProvider.hpp"

namespace o2::field
{
class MagneticField;
}

namespace o2::acts
{

/// Exposes the O2 magnetic field to ACTS.
///
/// Holds no field of its own: it reads the process-global field that
/// o2::base::Propagator::initFieldFromGRP() installs into
/// TGeoGlobalMagField::Instance() from the GRPMagField CCDB object. The
/// underlying object stays valid across field rescalings -- GRPGeomHelper
/// rescales the existing MagneticField in place rather than replacing it -- so
/// an adapter built once keeps returning up-to-date values.
///
/// Units: O2 works in kGauss and cm, ACTS in its native tesla/mm system, and the
/// conversion is applied here.
class MagneticFieldAdapter : public Acts::MagneticFieldProvider
{
 public:
  /// Use the field currently installed in TGeoGlobalMagField.
  /// \throw std::runtime_error if no field has been initialised
  MagneticFieldAdapter();

  /// Use an explicitly provided field.
  explicit MagneticFieldAdapter(o2::field::MagneticField* field);

  Cache makeCache(const Acts::MagneticFieldContext& mctx) const final;

  Acts::Result<Acts::Vector3> getField(const Acts::Vector3& position, Cache& cache) const final;

  const o2::field::MagneticField* getO2Field() const { return mField; }

 private:
  struct CacheImpl {
  };

  o2::field::MagneticField* mField = nullptr;
};

} // namespace o2::acts

#endif // ALICEO2_ACTS_MAGNETICFIELDADAPTER_H
