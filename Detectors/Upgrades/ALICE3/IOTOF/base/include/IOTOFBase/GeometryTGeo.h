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

  // Common i/o/f/bTOF
  static const char* getIOTOFVolPattern() { return sIOTOFVolumeName.c_str(); }

  // Inner TOF
  static const char* getITOFLayerPattern() { return sITOFLayerName.c_str(); }
  static const char* getITOFChipPattern() { return sITOFChipName.c_str(); }
  static const char* getITOFSensorPattern() { return sITOFSensorName.c_str(); }

  // Outer TOF
  static const char* getOTOFLayerPattern() { return sOTOFLayerName.c_str(); }
  static const char* getOTOFChipPattern() { return sOTOFChipName.c_str(); }
  static const char* getOTOFSensorPattern() { return sOTOFSensorName.c_str(); }

  // Forward TOF
  static const char* getFTOFLayerPattern() { return sFTOFLayerName.c_str(); }
  static const char* getFTOFChipPattern() { return sFTOFChipName.c_str(); }
  static const char* getFTOFSensorPattern() { return sFTOFSensorName.c_str(); }

  // Backward TOF
  static const char* getBTOFLayerPattern() { return sBTOFLayerName.c_str(); }
  static const char* getBTOFChipPattern() { return sBTOFChipName.c_str(); }
  static const char* getBTOFSensorPattern() { return sBTOFSensorName.c_str(); }

  static const char* composeSymNameIOTOF(int d)
  {
    return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::TF3).getName(), d);
  }

  // Inner TOF
  static const char* composeITOFSymNameLayer(int d, int layer);
  static const char* composeITOFSymNameChip(int d, int lr);
  static const char* composeITOFSymNameSensor(int d, int layer);

  // Outer TOF
  static const char* composeOTOFSymNameLayer(int d, int layer);
  static const char* composeOTOFSymNameChip(int d, int lr);
  static const char* composeOTOFSymNameSensor(int d, int layer);

  // Forward TOF
  static const char* composeFTOFSymNameLayer(int d, int layer);
  static const char* composeFTOFSymNameChip(int d, int lr);
  static const char* composeFTOFSymNameSensor(int d, int layer);

  // Backward TOF
  static const char* composeBTOFSymNameLayer(int d, int layer);
  static const char* composeBTOFSymNameChip(int d, int lr);
  static const char* composeBTOFSymNameSensor(int d, int layer);

 protected:
  // i/oTOF mother volume
  static std::string sIOTOFVolumeName;

  // Inner TOF
  static std::string sITOFLayerName;
  static std::string sITOFChipName;
  static std::string sITOFSensorName;

  // Outer TOF
  static std::string sOTOFLayerName;
  static std::string sOTOFChipName;
  static std::string sOTOFSensorName;

  // Forward TOF
  static std::string sFTOFLayerName;
  static std::string sFTOFChipName;
  static std::string sFTOFSensorName;

  // Backward TOF
  static std::string sBTOFLayerName;
  static std::string sBTOFChipName;
  static std::string sBTOFSensorName;

 private:
  static std::unique_ptr<o2::iotof::GeometryTGeo> sInstance;
};

} // namespace iotof
} // namespace o2
#endif