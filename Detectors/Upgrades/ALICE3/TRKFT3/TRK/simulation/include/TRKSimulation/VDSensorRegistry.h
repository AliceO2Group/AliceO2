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

#ifndef O2_TRK_VDSENSORREGISTRY_H
#define O2_TRK_VDSENSORREGISTRY_H

#include <string>
#include <vector>

namespace o2::trk
{

struct VDSensorDesc {
  enum class Region { Barrel,
                      Disk };
  enum class Type { Curved,
                    Plane,
  };
  std::string name; // sensor volume name
  int petal = -1;
  Region region = Region::Barrel;
  Type type = Type::Curved;
  int idx = -1; // layer or disk index
};

// Accessor (defined in VDGeometryBuilder.cxx)
std::vector<VDSensorDesc>& vdSensorRegistry();

// Utilities (defined in VDGeometryBuilder.cxx)
void clearVDSensorRegistry();
void registerSensor(const std::string& volName, int petal, VDSensorDesc::Region region, VDSensorDesc::Type type, int idx);

} // namespace o2::trk
#endif
