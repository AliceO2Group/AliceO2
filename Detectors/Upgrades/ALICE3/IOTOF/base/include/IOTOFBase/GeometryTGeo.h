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
  using DetMatrixCache::getMatrixL2G;
  using DetMatrixCache::getMatrixT2L;

  GeometryTGeo(bool build = false, int loadTrans = 0);
  void Build(int loadTrans);
  void fillMatrixCache(int mask);
  static GeometryTGeo* Instance();

  // Common i/o/f/bTOF
  static const char* getIOTOFVolPattern() { return sIOTOFVolumeName.c_str(); }

  // Inner TOF
  const int getITOFNumberOfChips() { return mNumberOfChipsIOTOF[0]; }
  static const char* getITOFLayerPattern() { return sITOFLayerName.c_str(); }
  static const char* getITOFStavePattern() { return sITOFStaveName.c_str(); }
  static const char* getITOFModulePattern() { return sITOFModuleName.c_str(); }
  static const char* getITOFChipPattern() { return sITOFChipName.c_str(); }
  static const char* getITOFSensorPattern() { return sITOFSensorName.c_str(); }

  // Outer TOF
  const int getOTOFNumberOfChips() { return mNumberOfChipsIOTOF[1]; }
  static const char* getOTOFLayerPattern() { return sOTOFLayerName.c_str(); }
  static const char* getOTOFStavePattern() { return sOTOFStaveName.c_str(); }
  static const char* getOTOFModulePattern() { return sOTOFModuleName.c_str(); }
  static const char* getOTOFChipPattern() { return sOTOFChipName.c_str(); }
  static const char* getOTOFSensorPattern() { return sOTOFSensorName.c_str(); }

  // Forward TOF
  const int getFTOFNumberOfChips() { return mNumberOfChipsFTOF; }
  static const char* getFTOFLayerPattern() { return sFTOFLayerName.c_str(); }
  static const char* getFTOFChipPattern() { return sFTOFChipName.c_str(); }
  static const char* getFTOFSensorPattern() { return sFTOFSensorName.c_str(); }

  // Backward TOF
  const int getBTOFNumberOfChips() { return mNumberOfChipsBTOF; }
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

  int getIOTOFFirstChipIndex(int lay) const;
  int getIOTOFLayer(int index) const;
  int getIOTOFChipIndex(int lay, int sta, int mod, int chip) const;
  bool getIOTOFChipId(int index, int& lay, int& sta, int& mod, int& chip) const;

  /// Get the transformation matrix of the SENSOR (not necessary the same as the chip)
  /// for a given chip 'index' by querying the TGeoManager
  TGeoHMatrix* extractMatrixSensor(int index) const;

  // sensor ref X and alpha
  void extractSensorXAlpha(int, float&, float&);

  // create matrix for tracking to local frame for IOTOF
  TGeoHMatrix& createT2LMatrix(int);

  TString getMatrixPath(int index) const;

  // cache for tracking frames
  void defineSensors();
  bool isTrackingFrameCached() const { return !mCacheRefX.empty(); }
  void fillTrackingFramesCache();

  float getSensorRefAlpha(int chipId) const
  {
    const int local = chipId;
    return mCacheRefAlpha[local];
  }

  float getSensorX(int chipId) const
  {
    const int local = chipId;
    return mCacheRefX[local];
  }

 protected:
  // Determine the number of active parts in the geometry
  int extractNumberOfStavesIOTOF(int lay) const;
  int extractNumberOfModulesIOTOF(int lay) const;
  int extractNumberOfChipsPerModuleIOTOF(int lay) const;
  int extractNumberOfChipsFTOF() const;
  int extractNumberOfChipsBTOF() const;

  // i/oTOF mother volume
  static std::string sIOTOFVolumeName;

  // Inner TOF
  static std::string sITOFLayerName;
  static std::string sITOFStaveName;
  static std::string sITOFModuleName;
  static std::string sITOFChipName;
  static std::string sITOFSensorName;

  // Outer TOF
  static std::string sOTOFLayerName;
  static std::string sOTOFStaveName;
  static std::string sOTOFModuleName;
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

  // Inner/outer TOF
  int mNumberOfStavesIOTOF[2];
  int mNumberOfModulesIOTOF[2];
  int mNumberOfChipsPerModuleIOTOF[2];
  int mNumberOfChipsPerStaveIOTOF[2];
  int mNumberOfChipsIOTOF[2];
  int mLastChipIndex[2];

  // Forward TOF
  int mNumberOfChipsFTOF;

  // Backward TOF
  int mNumberOfChipsBTOF;

  std::vector<int> sensors;
  std::vector<float> mCacheRefX;     /// cache for X of IOTOF
  std::vector<float> mCacheRefAlpha; /// cache for sensor ref alpha IOTOF

 private:
  static std::unique_ptr<o2::iotof::GeometryTGeo> sInstance;
};

} // namespace iotof
} // namespace o2
#endif
