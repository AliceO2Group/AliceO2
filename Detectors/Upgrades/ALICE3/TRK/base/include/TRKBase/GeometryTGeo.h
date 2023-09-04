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

#ifndef ALICEO2_TRK_GEOMETRYTGEO_H
#define ALICEO2_TRK_GEOMETRYTGEO_H

#include <memory>
#include <DetectorsCommonDataFormats/DetMatrixCache.h>

namespace o2
{
namespace trk
{
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  GeometryTGeo(bool build = false, int loadTrans = 0);
  void Build(int loadTrans);
  void fillMatrixCache(int mask);
  static GeometryTGeo* Instance();

  static const char* getTRKVolPattern() { return sVolumeName.c_str(); }
  static const char* getTRKLayerPattern() { return sLayerName.c_str(); }
  static const char* getTRKChipPattern() { return sChipName.c_str(); }
  static const char* getTRKSensorPattern() { return sSensorName.c_str(); }

  static const char* composeSymNameTRK(int d)
  {
    return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::TRK).getName(), d);
  }
  static const char* composeSymNameLayer(int d, int layer);
  static const char* composeSymNameChip(int d, int lr);
  static const char* composeSymNameSensor(int d, int layer);

 protected:
  static std::string sVolumeName;
  static std::string sLayerName;
  static std::string sChipName;
  static std::string sSensorName;

 private:
  static std::unique_ptr<o2::trk::GeometryTGeo> sInstance;
};

} // namespace trk
} // namespace o2
#endif