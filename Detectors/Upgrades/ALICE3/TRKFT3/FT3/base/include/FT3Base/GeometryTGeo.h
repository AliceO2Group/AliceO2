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

/// \file GeometryTGeo.h
/// \brief Definition of the GeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012
/// \author rafael.pezzi@cern.ch - adapted to PostLS4EndCaps 25/06/2020

#ifndef ALICEO2_FT3_GEOMETRYTGEO_H_
#define ALICEO2_FT3_GEOMETRYTGEO_H_

#include <memory>
#include "DetectorsCommonDataFormats/DetMatrixCache.h"
#include "DetectorsCommonDataFormats/DetID.h"
// #include "MathUtils/Utils.h"
// #include "Rtypes.h" // for Int_t, Double_t, Bool_t, UInt_t, etc

namespace o2
{
namespace ft3
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
    // get (create if needed) a unique instance of the object
    if (!sInstance) {
      sInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo(true, 0));
    }
    return sInstance.get();
  }

  // adopt the unique instance from external raw pointer (to be used only to read saved instance from file)
  static void adopt(GeometryTGeo* raw);

  int extractNumberOfDiscs(int dir);
  int extractNumberOfChips(int dir, int layer);
  int extractChipId(std::string const volName);
  void extractStaveChipId(std::string const volName, int& stave, int& chip);
  void extractChipIds(std::string const volName, int& direction, int& layer, int& stave, int& chip);

  int getChipIndex(int dir, int disc, int stave, int chip) const;
  // int getDisk(int index) const {return -1;} // TODO: implement this
  int getLayer(int chipIdx) const;
  std::string getMatrixPath(int direction, int layer, int stave, int chip) const;
  int getNumberOfChips() const { return mSize; }
  int getNumberOfLayers() const { return mNumberOfDiscs[0] + mNumberOfDiscs[1]; }
  int getNumberOfStaves(int absDisc) const { return mNumberOfStavesPerDisc[absDisc]; }
  int getSubDetID(int) const { return 2; }
  int getStave(int chipIdx) const;
  int getChipOnStave(int chipIdx) const;
  int getStaveIdxDisc(int absDisc) const { return mStaveIdxDisc[absDisc]; }
  int getChipIdxStave(int absStave) const { return mChipIdxStave[absStave]; }
  /// Exract FT3 parameters from TGeo

  bool isOwner() const { return mOwner; }
  void setOwner(bool v) { mOwner = v; }

  void Print(Option_t* opt = "") const;
  static const char* getFT3VolPattern() { return sVolumeName.c_str(); }
  static const char* getFT3InnerVolPattern() { return sInnerVolumeName.c_str(); }
  static const char* getFT3LayerPattern() { return sLayerName.c_str(); }
  static const char* getFT3ChipPattern() { return sChipName.c_str(); }
  static const char* getFT3SensorPattern() { return sSensorName.c_str(); }
  static const char* getFT3PassivePattern() { return sPassiveName.c_str(); }

  static const char* composeSymNameFT3(Int_t d) { return Form("%s_%d", o2::detectors::DetID(o2::detectors::DetID::FT3).getName(), d); }
  static const char* composeSymNameLayer(Int_t d, Int_t lr);
  static const char* composeSymNameChip(Int_t d, Int_t lr);
  static const char* composeSymNameSensor(Int_t d, Int_t lr);

 protected:
  static std::string sInnerVolumeName; ///< Mother inner volume name
  static std::string sVolumeName;      ///< Mother volume name
  static std::string sLayerName;       ///< Layer name
  static std::string sChipName;        ///< Chip name
  static std::string sSensorName;      ///< Sensor name
  static std::string sPassiveName;     ///< Passive material name

  std::vector<float> mCacheRefXDiscs;      /// cache for X of ML and OT
  std::vector<float> mCacheRefAlphaDiscs;  /// cache for sensor ref alpha ML and OT
  std::vector<int> mNumberOfDiscs;         ///< Number Discs per direction
  std::vector<int> mNumberOfStavesPerDisc; /// TODO; in principle redundant?
  std::vector<int> mStaveIdxDisc;          /// Index of first global stave Id for each disc
  std::vector<int> mChipIdxStave;          /// Index of first chup for each global stave
  std::vector<int> mNumberOfChipsPerDisc;  ///
  // std::vector<unsigned int> mChipIndexLayer;  ///< ID of first chip in the layer
  // std::vector<int> mChipStaveIds;

  bool mOwner = true; //! is it owned by the singleton?

 private:
  static std::unique_ptr<o2::ft3::GeometryTGeo> sInstance; ///< singleton instance
};

} // namespace ft3
} // namespace o2
#endif