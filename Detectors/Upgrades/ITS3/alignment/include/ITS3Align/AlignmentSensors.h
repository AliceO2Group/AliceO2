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

#ifndef O2_ITS3_ALIGNMENT_SENSORS_H
#define O2_ITS3_ALIGNMENT_SENSORS_H

#include "ITS3Align/AlignmentHierarchy.h"

namespace o2::its3::align
{

AlignableVolume::Ptr buildHierarchyITS(AlignableVolume::SensorMapping& sensorMap);
AlignableVolume::Ptr buildHierarchyIT3(AlignableVolume::SensorMapping& sensorMap);

class AlignableSensorITS final : public AlignableVolume
{
  using AlignableVolume::AlignableVolume;
  void defineMatrixL2G() final;
  void defineMatrixT2L() final;
  void computeJacobianL2T(const double* pos, Matrix66& jac) const final;
};

class AlignableSensorIT3 final : public AlignableVolume
{
  using AlignableVolume::AlignableVolume;
  void defineMatrixL2G() final;
  void defineMatrixT2L() final;
  void computeJacobianL2T(const double* pos, Matrix66& jac) const final;
};

} // namespace o2::its3::align

#endif
