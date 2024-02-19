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

#ifndef ALICEO2_MID_ALICE3_GEOMETRYTGEO_H
#define ALICEO2_MID_ALICE3_GEOMETRYTGEO_H

#include <DetectorsCommonDataFormats/DetMatrixCache.h>

namespace o2
{
namespace mi3
{
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  GeometryTGeo(bool build = false, int loadTrans = 0);
  void Build(int loadTrans);
  void fillMatrixCache(int mask);
  static GeometryTGeo* Instance();

  static const char* getMIDVolPattern() { return sVolumeName.c_str(); }
  static const char* getMIDSensorPattern() { return sSensorName.c_str(); }

  static const char* composeSymNameMID()
  {
    return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::MI3).getName(), 0);
  }
  static const char* composeSymNameSensor(); // A single sensor for the moment

 protected:
  static std::string sVolumeName;
  static std::string sSensorName;

 private:
  static std::unique_ptr<o2::mi3::GeometryTGeo> sInstance;
};
} // namespace mi3
} // namespace o2
#endif
