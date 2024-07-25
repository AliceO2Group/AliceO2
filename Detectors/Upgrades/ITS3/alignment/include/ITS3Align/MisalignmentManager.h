// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ITS3_MISALIGNMENTMANAGER_H_
#define ITS3_MISALIGNMENTMANAGER_H_

#include "Math/Transform3D.h"
#include "Math/Translation3D.h"
#include "Math/Rotation3D.h"
#include "Math/EulerAngles.h"
#include "Math/PositionVector3D.h"
#include "TGeoMatrix.h"

#include <filesystem>

namespace o2::its3::align
{

/// Collection of static functions and types to perform misalignment studies
struct MisalignmentManager {
  using Vector3D = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::DefaultCoordinateSystemTag>;
  using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::DefaultCoordinateSystemTag>;
  using Trans3D = ROOT::Math::Translation3DF;
  using Rot3D = ROOT::Math::Rotation3D;
  using Euler3D = ROOT::Math::EulerAngles;
  using Trafo3D = ROOT::Math::Transform3DF;

  static void misalignHits();

  static void createBackup(const std::filesystem::path& src, const std::filesystem::path& dest);

  static std::string appendStem(const std::string& filename, const std::string& add);

  static std::vector<std::string> split(const std::string& s, char delimiter = '/');

  static void navigate(const std::string& path);

  static std::string composePathSensor(int sensor);

  static void applyGlobalMatrixVolume(const std::string& path, const TGeoHMatrix& globalMatrix);
};

} // namespace o2::its3::align

#endif
