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
#include "DetectorsCommonDataFormats/DetID.h"
#include "TRKBase/TRKBaseParam.h"

namespace o2
{
namespace trk
{
class GeometryTGeo : public o2::detectors::DetMatrixCache
{
 public:
  using Mat3D = o2::math_utils::Transform3D;
  using DetMatrixCache::getMatrixL2G;
  using DetMatrixCache::getMatrixT2GRot;
  using DetMatrixCache::getMatrixT2L;
  // this method is not advised for ITS: for barrel detectors whose tracking frame is just a rotation
  // it is cheaper to use T2GRot
  using DetMatrixCache::getMatrixT2G;
  GeometryTGeo(bool build = false, int loadTrans = 0);
  ~GeometryTGeo();
  void Build(int loadTrans);
  void fillMatrixCache(int mask);
  static GeometryTGeo* Instance()
  {
    if (!sInstance) {
      sInstance = std::make_unique<GeometryTGeo>(true, 0);
    }
    return sInstance.get();
  };

  static const char* getFT3VolPattern() { return sFT3VolumeName.c_str(); }
  static const char* getFT3InnerVolPattern() { return sFT3InnerVolumeName.c_str(); }
  static const char* getFT3LayerPattern() { return sFT3LayerName.c_str(); }
  static const char* getFT3ChipPattern() { return sFT3ChipName.c_str(); }
  static const char* getFT3PassivePattern() { return sFT3PassiveName.c_str(); }
  static const char* getFT3SensorPattern() { return sFT3SensorName.c_str(); }
  static const char* getTRKVolPattern() { return sVolumeName.c_str(); }
  static const char* getTRKServiceVolPattern() { return sServiceVolName.c_str(); }
  static const char* getTRKLayerPattern() { return sLayerName.c_str(); }
  static const char* getTRKPetalAssemblyPattern() { return sPetalAssemblyName.c_str(); }
  static const char* getTRKPetalPattern() { return sPetalName.c_str(); }
  static const char* getTRKPetalDiskPattern() { return sPetalDiskName.c_str(); }
  static const char* getTRKPetalLayerPattern() { return sPetalLayerName.c_str(); }
  static const char* getTRKStavePattern() { return sStaveName.c_str(); }
  static const char* getTRKHalfStavePattern() { return sHalfStaveName.c_str(); }
  static const char* getTRKModulePattern() { return sModuleName.c_str(); }
  static const char* getTRKChipPattern() { return sChipName.c_str(); }
  static const char* getTRKSensorPattern() { return sSensorName.c_str(); }
  static const char* getTRKDeadzonePattern() { return sDeadzoneName.c_str(); }
  static const char* getTRKMetalStackPattern() { return sMetalStackName.c_str(); }

  static const char* getTRKWrapVolPattern() { return sWrapperVolumeName.c_str(); }

  int getNumberOfChips() const { return mSize; }

  /// Determines the number of active parts in the Geometry
  int extractNumberOfDisksMLOT(int dir) const;
  int extractNumberOfLayersMLOT();
  int extractNumberOfLayersVD() const;
  int extractNumberOfPetalsVD() const;
  int extractNumberOfActivePartsVD() const;
  int extractNumberOfDisksVD() const;
  int extractNumberOfChipsPerPetalVD() const;
  int extractNumberOfStavesMLOT(int lay) const;
  int extractNumberOfHalfStavesMLOT(int lay) const;
  int extractNumberOfModulesMLOT(int lay) const;
  int extractNumberOfChipsMLOT(int lay) const;

  /// Extract number following the prefix in the name string
  void extractChipIdsFT3(std::string const volName, int& layer, int& stave, int& chip) const;
  int extractVolumeCopy(const char* name, const char* prefix) const;

  int getNumberOfLayersMLOT() const { return mNumberOfLayersMLOT; }
  int getNumberOfActivePartsVD() const { return mNumberOfActivePartsVD; }
  int getNumberOfHalfStaves(int lay) const { return mNumberOfHalfStaves[lay]; }

  bool isOwner() const { return mOwner; }
  void setOwner(bool v) { mOwner = v; }

  void Print(Option_t* opt = "") const;
  void PrintChipID(int index, int subDetID, int petalcase, int disk, int lay, int stave, int halfstave, int mod, int chip) const;

  int getSubDetID(int index) const;
  int getPetalCase(int index) const;
  int getDisk(int index) const;
  int getLayer(int index) const;    ///< local layer index within the sub-detector (0-based per VD/MLOT)
  int getLayerTRK(int index) const; ///< global layer index across the full TRK (VD layers 0..nVD-1, MLOT layers nVD..nTotal-1)
  int getStave(int index) const;
  int getHalfStave(int index) const;
  int getModule(int index) const;
  int getChip(int index) const;

  void defineMLOTSensors();
  int getBarrelLayer(int) const;

  // sensor ref X and alpha for ML & OT
  void extractSensorXAlphaMLOT(int, float&, float&);

  // cache for tracking frames (ML & OT)
  bool isTrackingFrameCachedMLOT() const { return !mCacheRefXMLOT.empty(); }
  void fillTrackingFramesCacheMLOT();

  float getSensorRefAlphaMLOT(int chipId) const
  {
    if (getSubDetID(chipId) == 0) {
      LOG(error) << "getSensorRefAlphaMLOT(): VD layers are not supported yet! chipID = " << chipId
                 << "please provide chipId for ML/OT! ";
      return std::numeric_limits<float>::quiet_NaN();
    }
    const int local = chipId - getNumberOfActivePartsVD();
    return mCacheRefAlphaMLOT[local];
  }

  float getSensorXMLOT(int chipId) const
  {
    if (getSubDetID(chipId) == 0) {
      LOG(error) << "getSensorXMLOT(): VD layers are not supported yet! chipID = " << chipId
                 << "please provide chipId for ML/OT! ";
      return std::numeric_limits<float>::quiet_NaN();
    }
    const int local = chipId - getNumberOfActivePartsVD();
    return mCacheRefXMLOT[local];
  }

  // create matrix for tracking to local frame for MLOT
  TGeoHMatrix& createT2LMatrixMLOT(int);

  /// This routine computes the chip index number from the subDetID, petal, disk, layer, stave /// TODO: retrieve also from chip when chips will be available
  /// This routine computes the chip index number from the subDetID, petal, disk, layer, stave, half stave, module, chip
  /// The subdetectors are numbers as follows: 0: VD; 1: ML+OT barrels; 2: ML+OT discs
  /// \param int subDetID The subdetector ID, 0 for VD, 1 for MLOT
  /// \param int petalcase The petal case number for VD, from 0 to 3
  /// \param int disk The disk number for VD or OT (VD 0-6 if present; OT 0-12 (18 for V1 geometry)
  /// \param int lay The layer number. Starting from 0 both for VD and MLOT
  /// \param int stave The stave number for MLOT. Starting from 0
  /// \param int halfstave The half stave number for MLOT. Can be 0 or 1
  /// \param int module The module number for MLOT, from 0 to 10 (or 20)
  /// \param int chip The chip number for MLOT, from 0 to 8
  unsigned short getChipIndex(int subDetID, int petalcase, int disk, int lay, int stave, int halfstave, int mod, int chip) const;

  /// This routine computes the chip index number from the subDetID, volume, layer, stave, half stave, module, chip
  /// \param int subDetID The subdetector ID, 0 for VD, 1 for MLOT
  /// \param int volume is needed only with the current configuration for VD where each single element is a volume. // TODO: when the geometry naming scheme will be changed, change this method
  /// \param int lay The layer number for the MLOT. In the current configuration for VD this is not needed. // TODO: when the geometry naming scheme will be changed, change this method
  /// \param int stave The stave number in each layer for MLOT. Starting from 0.
  /// \param int halfstave The half stave number for MLOT. Can be 0 or 1
  /// \param int module The module number for MLOT, from 0 to 10 (or 20)
  /// \param int chip The chip number for MLOT, from 0 to 8
  unsigned short getChipIndex(int subDetID, int volume, int lay, int stave, int halfstave, int mod, int chip) const;

  /// This routine computes subDetID, petal, disk, layer, stave, half stave, module, chip, given the chip index number
  /// \param int index The chip index number, starting from 0
  /// \param int subDetID The subdetector ID, 0 for VD, 1 for MLOT
  /// \param int petalcase The petal case number for VD, from 0 to 3
  /// \param int disk The disk number for VD, from 0 to 5
  /// \param int lay The layer number. Starting from 0 both for VD and MLOT
  /// \param int stave The stave number for MLOT. Starting from 0
  /// \param int halfstave The half stave number for MLOT. Can be 0 or 1
  /// \param int module The module number for MLOT, from 0 to 10 (or 20)
  /// \param int chip The chip number for MLOT, from 0 to 8
  bool getChipID(int index, int& subDetID, int& petalcase, int& disk, int& lay, int& stave, int& halfstave, int& mod, int& chip) const;

  unsigned short getLastChipIndex(int lay) const { return mLastChipIndex[lay]; }
  unsigned short getFirstChipIndex(int lay, int petalcase, int subDetID) const
  {
    /// Get the first chip index of the active petal (VD) or layer (MLOT)
    if (subDetID == 0) { // VD
      return (petalcase == 0) ? 0 : mLastChipIndexVD[petalcase - 1] + 1;
    } else if (subDetID == 1) { // MLOT
      return mLastChipIndex[lay + mNumberOfPetalsVD - 1] + 1;
    } else if (subDetID == 2) {
      return mFirstChipIndexMLOTDisc[lay];
    }
    return -1; // not found
  }

  /// Get the transformation matrix of the SENSOR (not necessary the same as the chip)
  /// for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* extractMatrixSensor(int index) const;

  TString getMatrixPath(int index) const;

#ifdef ENABLE_UPGRADES
  static const char* composeSymNameTRK(int d)
  {
    return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::TRK).getName(), d);
  }
#endif

  static const char* composeSymNameLayer(int d, int layer);
  static const char* composeSymNameLayerFT3(int dir, int layer);
  static const char* composeSymNameStave(int d, int layer);
  static const char* composeSymNameModule(int d, int layer);
  static const char* composeSymNameChip(int d, int layer);
  static const char* composeSymNameSensor(int d, int layer);

 protected:
  static constexpr int MAXLAYERS = 25; ///< max number of active layers

  static std::string sVolumeName;
  static std::string sServiceVolName;
  static std::string sLayerName;
  static std::string sPetalAssemblyName;
  static std::string sPetalName;
  static std::string sPetalDiskName;
  static std::string sPetalLayerName;
  static std::string sStaveName;
  static std::string sHalfStaveName;
  static std::string sModuleName;
  static std::string sChipName;
  static std::string sSensorName;
  static std::string sDeadzoneName;
  static std::string sMetalStackName;

  static std::string sWrapperVolumeName; ///< Wrapper volume name, not implemented at the moment

  static std::string sFT3InnerVolumeName; ///< Mother inner volume name
  static std::string sFT3VolumeName;      ///< Mother volume name
  static std::string sFT3LayerName;       ///< Layer name
  static std::string sFT3ChipName;        ///< Chip name
  static std::string sFT3PassiveName;     ///< Passive material name
  static std::string sFT3SensorName;      ///< Sensor name

  Int_t mNumberOfLayersMLOT;                   ///< number of layers
  Int_t mNumberOfDisksMLOT;                    ///< number of ML/OT disks (12 for v3)
  Int_t mNumberOfActivePartsVD;                ///< number of layers
  Int_t mNumberOfLayersVD;                     ///< number of layers
  Int_t mNumberOfPetalsVD;                     ///< number of Petals = chip in each VD layer
  Int_t mNumberOfDisksVD;                      ///< number of Disks = 6
  std::vector<int> mNumberOfStaves;            ///< Number Of Staves per layer in ML/OT barrels
  std::vector<int> mNumberOfStavesMLOTDDiscs;  ///< Number Of Staves per layer in ML/OT discs
  std::vector<int> mNumberOfHalfStaves;        ///< Number Of Half staves in each stave of the layer in ML/OT
  std::vector<int> mNumberOfModules;           ///< Number Of Modules per stave (half stave) in ML/OT
  std::vector<int> mNumberOfChips;             ///< number of chips per module in ML/OT
  std::vector<int> mNumberOfChipsPerLayerVD;   ///< number of chips per layer VD ( =  number of petals)
  std::vector<int> mNumberOfChipsPerLayerMLOT; ///< number of chips per layer MLOT
  std::vector<int> mNumberOfChipPerDiskMLOT;   ///< number of chips per disc in MLOT
  std::vector<int> mNumbersOfChipPerDiskVD;    ///< numbersOfChipPerDiskVD
  std::vector<int> mNumberOfChipsPerPetalVD;   ///< numbersOfChipPerPetalVD
  // std::vector<int> mNumberOfChipsPerStave;     ///< number of chips per stave in ML/OT
  // std::vector<int> mNumberOfChipsPerHalfStave; ///< number of chips per half stave in ML/OT
  // std::vector<int> mNumberOfChipsPerModule; ///< number of chips per module in ML/OT
  std::vector<unsigned short> mLastChipIndex;     ///< max ID of the detector in the petal(VD) or layer(MLOT)
  std::vector<unsigned short> mLastChipIndexVD;   ///< max ID of the detector in the layer for the VD
  // std::vector<unsigned short> mLastChipIndexMLOT; ///< max ID of the detector in the layer for the MLOT
  std::vector<unsigned short> mFirstChipIndexMLOTDisc; ///< ID of the first sensor chip in the layer for the MLOT; array size is one larger than the number of disks; last element equals nChips+1
  std::vector<int> mFirstStaveIndexDisc;               ///< Index of first stave (abs ID) in each MLOT Disc
  std::vector<int> mFirstChipIndexStave;               ///< Index of first chip on stave (Discs)
  std::array<char, MAXLAYERS> mLayerToWrapper; ///< Layer to wrapper correspondence, not implemented yet

  bool mOwner = true; //! is it owned by the singleton?

  std::vector<int> sensorsMLOT;
  std::vector<float> mCacheRefXMLOT;     /// cache for X of ML and OT
  std::vector<float> mCacheRefAlphaMLOT; /// cache for sensor ref alpha ML and OT

  eMLOTLayout mLayoutMLOT; // ML and OT detector layout design

 private:
  static std::unique_ptr<o2::trk::GeometryTGeo> sInstance;
};

} // namespace trk
} // namespace o2
#endif
