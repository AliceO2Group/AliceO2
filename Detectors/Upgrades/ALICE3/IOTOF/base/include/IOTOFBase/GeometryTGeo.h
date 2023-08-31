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

#ifndef ALICEO2_IOTOF_GEOMETRYTGEO_H
#define ALICEO2_IOTOF_GEOMETRYTGEO_H

#include <memory>
#include <DetectorsCommonDataFormats/DetMatrixCache.h>

namespace o2
{
namespace iotof
{
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  GeometryTGeo(bool build = false, int loadTrans = 0);
  void Build(int loadTrans);
  void fillMatrixCache(int mask);
  static GeometryTGeo* Instance();

  // Inner TOF
  static const char* getITOFVolPattern() { return sITOFVolumeName.c_str(); }
  static const char* getITOFLayerPattern() { return sITOFLayerName.c_str(); }
  static const char* getITOFChipPattern() { return sITOFChipName.c_str(); }
  static const char* getITOFSensorPattern() { return sITOFSensorName.c_str(); }

  // Outer TOF
  static const char* getOTOFVolPattern() { return sOTOFVolumeName.c_str(); }
  static const char* getOTOFLayerPattern() { return sOTOFLayerName.c_str(); }
  static const char* getOTOFChipPattern() { return sOTOFChipName.c_str(); }
  static const char* getOTOFSensorPattern() { return sOTOFSensorName.c_str(); }

  static const char* composeSymNameITOF(int d)
  {
    return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::ITF).getName(), d);
  }

  static const char* composeSymNameOTOF(int d)
  {
    return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::OTF).getName(), d);
  }

  // Inner TOF
  static const char* composeITOFSymNameLayer(int d, int layer);
  static const char* composeITOFSymNameChip(int d, int lr);
  static const char* composeITOFSymNameSensor(int d, int layer);

  // Outer TOF
  static const char* composeOTOFSymNameLayer(int d, int layer);
  static const char* composeOTOFSymNameChip(int d, int lr);
  static const char* composeOTOFSymNameSensor(int d, int layer);

 protected:
  // Inner TOF
  static std::string sITOFVolumeName;
  static std::string sITOFLayerName;
  static std::string sITOFChipName;
  static std::string sITOFSensorName;

  // Outer TOF
  static std::string sOTOFVolumeName;
  static std::string sOTOFLayerName;
  static std::string sOTOFChipName;
  static std::string sOTOFSensorName;

 private:
  static std::unique_ptr<o2::iotof::GeometryTGeo> sInstance;
};

} // namespace iotof
} // namespace o2
#endif