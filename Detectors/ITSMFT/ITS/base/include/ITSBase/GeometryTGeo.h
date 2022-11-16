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

#ifndef ALICEO2_ITS_GEOMETRYTGEO_H_
#define ALICEO2_ITS_GEOMETRYTGEO_H_

#include <TGeoMatrix.h> // for TGeoHMatrix
#include <TObject.h>    // for TObject
#include <array>
#include <string>
#include <vector>
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"
#include "Rtypes.h" // for Int_t, Double_t, Bool_t, UInt_t, etc

class TGeoPNEntry;

namespace o2
{
namespace its
{
/// GeometryTGeo is a simple interface class to TGeoManager. It is used in the simulation
/// and reconstruction in order to query the TGeo ITS geometry.
/// RS: In order to preserve the static character of the class but make it dynamically access
/// geometry, we need to check in every method if the structures are initialized. To be converted
/// to singleton at later stage.

class GeometryTGeo : public o2::itsmft::GeometryTGeo
{
 public:
  typedef o2::math_utils::Transform3D Mat3D;
  using DetMatrixCache::getMatrixL2G;
  using DetMatrixCache::getMatrixT2GRot;
  using DetMatrixCache::getMatrixT2L;
  // this method is not advised for ITS: for barrel detectors whose tracking frame is just a rotation
  // it is cheaper to use T2GRot
  using DetMatrixCache::getMatrixT2G;

  static GeometryTGeo* Instance()
  {
    // get (create if needed) a unique instance of the object
#ifdef GPUCA_STANDALONE
    return nullptr; // TODO: DR: Obviously wrong, but to make it compile for now
#else
    if (!sInstance) {
      sInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo(true, 0));
    }
    return sInstance.get();
#endif
  }

  // adopt the unique instance from external raw pointer (to be used only to read saved instance from file)
  static void adopt(GeometryTGeo* raw);

  // constructor
  // ATTENTION: this class is supposed to behave as a singleton, but to make it root-persistent
  // we must define public default constructor.
  // NEVER use it, it will throw exception if the class instance was already created
  // Use GeometryTGeo::Instance() instead
  GeometryTGeo(bool build = kFALSE, int loadTrans = 0
               /*o2::base::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, // default transformations to load
           o2::math_utils::TransformType::T2G,
           o2::math_utils::TransformType::L2G)*/
  );

  /// Default destructor
  ~GeometryTGeo() override;

  GeometryTGeo(const GeometryTGeo& src) = delete;
  GeometryTGeo& operator=(const GeometryTGeo& geom) = delete;

  // implement filling of the matrix cache
  using o2::itsmft::GeometryTGeo::fillMatrixCache;
  void fillMatrixCache(int mask) override;

  // cache parameters of sensors tracking frames
  void fillTrackingFramesCache();

  /// Exract ITS parameters from TGeo
  void Build(int loadTrans = 0) override;

  int getNumberOfChipRowsPerModule(int lay) const { return mNumberOfChipRowsPerModule[lay]; }
  int getNumberOfChipColsPerModule(int lay) const
  {
    return mNumberOfChipRowsPerModule[lay] ? mNumberOfChipsPerModule[lay] / mNumberOfChipRowsPerModule[lay] : -1;
  }

  int getNumberOfChipsPerModule(int lay) const { return mNumberOfChipsPerModule[lay]; }
  int getNumberOfChipsPerHalfStave(int lay) const { return mNumberOfChipsPerHalfStave[lay]; }
  int getNumberOfChipsPerStave(int lay) const { return mNumberOfChipsPerStave[lay]; }
  int getNumberOfChipsPerHalfBarrel(int lay) const { return mNumberOfChipsPerHalfBarrel[lay]; }
  int getNumberOfChipsPerLayer(int lay) const { return mNumberOfChipsPerLayer[lay]; }
  int getNumberOfModules(int lay) const { return mNumberOfModules[lay]; }
  int getNumberOfHalfStaves(int lay) const { return mNumberOfHalfStaves[lay]; }
  int getNumberOfStaves(int lay) const { return mNumberOfStaves[lay]; }
  int getNumberOfHalfBarrels() const { return mNumberOfHalfBarrels; }
  int getNumberOfLayers() const { return mNumberOfLayers; }
  int getChipIndex(int lay, int detInLay) const { return getFirstChipIndex(lay) + detInLay; }
  /// This routine computes the chip index number from the layer, stave, and chip number in stave
  /// \param int lay The layer number. Starting from 0.
  /// \param int hba The halfbarrel number. Starting from 0
  /// \param int sta The stave number. Starting from 0
  /// \param int chipInStave The chip number in the stave. Starting from 0
  int getChipIndex(int lay, int hba, int sta, int detInSta) const;

  /// This routine computes the chip index number from the layer, stave, substave and chip number
  /// in substave
  /// \param int lay The layer number. Starting from 0.
  /// \param int hba The halfbarrel number. Starting from 0
  /// \param int sta The stave number. Starting from 0
  /// \param int substa The substave number. Starting from 0
  /// \param int chipInSStave The chip number in the sub stave. Starting from 0
  int getChipIndex(int lay, int hba, int sta, int subSta, int detInSubSta) const;

  /// This routine computes the chip index number from the layer,stave, substave module and
  /// chip number in module.
  /// \param int lay The layer number. Starting from 0.
  /// \param int hba The halfbarrel number. Starting from 0
  /// \param int sta The stave number. Starting from 0
  /// \param int substa The substave number. Starting from 0
  /// \param int module The module number ...
  /// \param int chipInSStave The chip number in the module. Starting from 0
  int getChipIndex(int lay, int hba, int sta, int subSta, int md, int detInMod) const;

  /// This routine computes the layer, stave, substave, module and chip number
  /// given the chip index number
  /// \param int index The chip index number, starting from zero.
  /// \param int lay The layer number. Starting from 0
  /// \param int sta The stave number. Starting from 0
  /// \param int ssta The halfstave number. Starting from 0
  /// \param int mod The module number. Starting from 0
  /// \param int chip The detector number. Starting from 0
  bool getChipId(int index, int& lay, int& sta, int& ssta, int& mod, int& chip) const;

  /// This routine computes the layer, half barrel, stave, substave,
  /// module and chip number given the chip index number
  /// \param int index The chip index number, starting from zero.
  /// \param int lay The layer number. Starting from 0
  /// \param int hba The half barrel number. Starting from 0
  /// \param int sta The stave number. Starting from 0
  /// \param int ssta The halfstave number. Starting from 0
  /// \param int mod The module number. Starting from 0
  /// \param int chip The detector number. Starting from 0
  bool getChipId(int index, int& lay, int& hba, int& sta, int& ssta, int& mod, int& chip) const;

  /// Get chip layer, from 0
  int getLayer(int index) const;

  /// Get chip half barrel, from 0
  int getHalfBarrel(int index) const;

  /// Get chip stave, from 0
  int getStave(int index) const;

  /// Get chip substave id in stave, from 0
  int getHalfStave(int index) const;

  /// Get chip module id in substave, from 0
  int getModule(int index) const;

  /// Get chip number within layer, from 0
  int getChipIdInLayer(int index) const;

  /// Get chip number within stave, from 0
  int getChipIdInStave(int index) const;

  /// Get chip number within stave, from 0
  int getChipIdInHalfStave(int index) const;

  /// Get chip number within module, from 0
  int getChipIdInModule(int index) const;

  int getLastChipIndex(int lay) const { return mLastChipIndex[lay]; }
  int getFirstChipIndex(int lay) const { return (lay == 0) ? 0 : mLastChipIndex[lay - 1] + 1; }
  const char* getSymbolicName(int index) const
  {
    /// return symbolic name of sensor
    return o2::base::GeometryManager::getSymbolicName(getDetID(), index);
  }

  const char* getSymbolicName(int lay, int hba, int sta, int det) const
  {
    /// return symbolic name of sensor
    return getSymbolicName(getChipIndex(lay, hba, sta, det));
  }

  /// Get the transformation matrix for a given chip (NOT A SENSOR!!!) 'index' by quering the TGeoManager
  TGeoHMatrix* getMatrix(int index) const { return o2::base::GeometryManager::getMatrix(getDetID(), index); }
  TGeoHMatrix* getMatrix(int lay, int hba, int sta, int sens) const { return getMatrix(getChipIndex(lay, hba, sta, sens)); }
  bool getOriginalMatrix(int index, TGeoHMatrix& m) const
  {
    /// Get the original (ideal geometry) TGeo matrix for a given chip identified by 'index'
    /// The method is slow, so it should be used with great care (for caching only)
    return o2::base::GeometryManager::getOriginalMatrix(getDetID(), index, m);
  }

  bool getOriginalMatrix(int lay, int hba, int sta, int det, TGeoHMatrix& m) const
  {
    /// Get the original (ideal geometry) TGeo matrix for a given chip identified by 'index'
    /// The method is slow, so it should be used with great care (for caching only)
    return getOriginalMatrix(getChipIndex(lay, hba, sta, det), m);
  }

  const Mat3D& getMatrixT2L(int lay, int hba, int sta, int det) const { return getMatrixT2L(getChipIndex(lay, hba, sta, det)); }
  const Mat3D& getMatrixSensor(int index) const { return getMatrixL2G(index); }
  const Mat3D& getMatrixSensor(int lay, int hba, int sta, int det)
  {
    // get positioning matrix of the sensor, alias to getMatrixL2G
    return getMatrixSensor(getChipIndex(lay, hba, sta, det));
  }

  const Rot2D& getMatrixT2GRot(int lay, int hba, int sta, int sens)
  {
    /// get matrix for tracking to global frame transformation
    return getMatrixT2GRot(getChipIndex(lay, hba, sta, sens));
  }

  bool isTrackingFrameCached() const { return !mCacheRefX.empty(); }
  void getSensorXAlphaRefPlane(int index, float& x, float& alpha) const
  {
    x = getSensorRefX(index);
    alpha = getSensorRefAlpha(index);
  }

  float getSensorRefX(int isn) const { return mCacheRefX[isn]; }
  float getSensorRefAlpha(int isn) const { return mCacheRefAlpha[isn]; }
  // Attention: these are transformations wrt sensitive volume!
  void localToGlobal(int index, const double* loc, double* glob);

  void localToGlobal(int lay, int sta, int det, const double* loc, double* glob);

  void globalToLocal(int index, const double* glob, double* loc);

  void globalToLocal(int lay, int sta, int det, const double* glob, double* loc);

  void localToGlobalVector(int index, const double* loc, double* glob);

  void globalToLocalVector(int index, const double* glob, double* loc);

  void Print(Option_t* opt = "") const;

  static const char* getITSVolPattern() { return sVolumeName.c_str(); }
  static const char* getITSLayerPattern() { return sLayerName.c_str(); }
  static const char* getITSHalfBarrelPattern() { return sHalfBarrelName.c_str(); }
  static const char* getITSWrapVolPattern() { return sWrapperVolumeName.c_str(); }
  static const char* getITSStavePattern() { return sStaveName.c_str(); }
  static const char* getITSHalfStavePattern() { return sHalfStaveName.c_str(); }
  static const char* getITSModulePattern() { return sModuleName.c_str(); }
  static const char* getITSChipPattern() { return sChipName.c_str(); }
  static const char* getITSSensorPattern() { return sSensorName.c_str(); }
  static void setITSVolPattern(const char* nm) { sVolumeName = nm; }
  static void setITSLayerPattern(const char* nm) { sLayerName = nm; }
  static void setITSHalfBarrelPattern(const char* nm) { sHalfBarrelName = nm; }
  static void setITSWrapVolPattern(const char* nm) { sWrapperVolumeName = nm; }
  static void setITSStavePattern(const char* nm) { sStaveName = nm; }
  static void setITSHalfStavePattern(const char* nm) { sHalfStaveName = nm; }
  static void setITSModulePattern(const char* nm) { sModuleName = nm; }
  static void setITSChipPattern(const char* nm) { sChipName = nm; }
  static void setITSSensorPattern(const char* nm) { sSensorName = nm; }

  static const char* getITS3LayerPattern() { return sLayerNameITS3.c_str(); }
  static const char* getITS3HalfBarrelPattern() { return sHalfBarrelNameITS3.c_str(); }
  static const char* getITS3StavePattern() { return sStaveNameITS3.c_str(); }
  static const char* getITS3HalfStavePattern() { return sHalfStaveNameITS3.c_str(); }
  static const char* getITS3ModulePattern() { return sModuleNameITS3.c_str(); }
  static const char* getITS3ChipPattern() { return sChipNameITS3.c_str(); }
  static const char* getITS3SensorPattern() { return sSensorNameITS3.c_str(); }
  /// sym name of the layer
  static const char* composeSymNameITS(bool isITS3 = false);
  /// sym name of the layer
  static const char* composeSymNameLayer(int lr, bool isITS3 = false);

  /// Sym name of the half barrel at given layer
  static const char* composeSymNameHalfBarrel(int lr, int hba, bool isITS3 = false);

  /// Sym name of the stave at given layer
  static const char* composeSymNameStave(int lr, int hba, int sta, bool isITS3 = false);

  /// Sym name of the stave at given layer/halfbarrel
  static const char* composeSymNameHalfStave(int lr, int hba, int sta, int ssta, bool isITS3 = false);

  /// Sym name of the substave at given layer/halfbarrel/stave
  static const char* composeSymNameModule(int lr, int hba, int sta, int ssta, int mod, bool isITS3 = false);

  /// Sym name of the chip in the given layer/halfbarrel/stave/substave/module
  static const char* composeSymNameChip(int lr, int hba, int sta, int ssta, int mod, int chip, bool isITS3 = false);

 protected:
  /// Get the transformation matrix of the SENSOR (not necessary the same as the chip)
  /// for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* extractMatrixSensor(int index) const;

  // create matrix for transformation from sensor local frame to global one
  TGeoHMatrix& createT2LMatrix(int isn);

  // get sensor tracking frame alpha and
  void extractSensorXAlpha(int isn, float& x, float& alp);

  /// This routine computes the layer number a given the chip index
  /// \param int index The chip index number, starting from zero.
  /// \param int indexInLr The chip index inside a layer, starting from zero.
  /// \param int lay The layer number. Starting from 0.
  bool getLayer(int index, int& lay, int& index2) const;

  /// Determines the number of chips per module on the (sub)stave in the Geometry
  /// Also extract the layout: span of module centers in Z and X
  /// \param lay: layer number from 0
  int extractNumberOfChipsPerModule(int lay, int& nrow) const;

  /// Determines the number of halfbarrels in the layer
  /// \param lay: layer number, starting from 0
  int extractNumberOfHalfBarrels() const;

  /// Determines the number of layers in the Geometry
  /// \param lay: layer number, starting from 0
  int extractNumberOfStaves(int lay) const;

  /// Determines the number of substaves in the stave of the layer
  /// \param lay: layer number, starting from 0
  int extractNumberOfHalfStaves(int lay) const;

  /// Determines the number of modules in substave in the stave of the layer
  /// \param lay: layer number, starting from 0
  /// For the setup w/o modules defined the module and the stave or the substave is the same thing
  /// Legacy method, keep it just in case...
  int extractNumberOfModules(int lay) const;

  /// Determines the layer detector type the Geometry and
  /// returns the detector type id for the layer
  /// \param lay: layer number from 0
  int extractLayerChipType(int lay) const;

  /// Determines the number of layers in the Geometry
  int extractNumberOfLayers();

  /// Extract number following the prefix in the name string
  int extractVolumeCopy(const char* name, const char* prefix) const;

  TGeoPNEntry* getPNEntry(int index) const
  {
    /// Get a pointer to the TGeoPNEntry of a chip identified by 'index'
    /// Returns NULL in case of invalid index, missing TGeoManager or invalid symbolic name
    return o2::base::GeometryManager::getPNEntry(getDetID(), index);
  }

 protected:
  static constexpr int MAXLAYERS = 15; ///< max number of active layers

  Int_t mNumberOfLayers;                        ///< number of layers
  Int_t mNumberOfHalfBarrels;                   ///< number of halfbarrels
  std::vector<int> mNumberOfStaves;             ///< number of staves/layer(layer)
  std::vector<int> mNumberOfHalfStaves;         ///< the number of substaves/stave(layer)
  std::vector<int> mNumberOfModules;            ///< number of modules/substave(layer)
  std::vector<int> mNumberOfChipsPerModule;     ///< number of chips per module (group of chips on substaves)
  std::vector<int> mNumberOfChipRowsPerModule;  ///< number of chips rows per module (relevant for OB modules)
  std::vector<int> mNumberOfChipsPerHalfStave;  ///< number of chips per substave
  std::vector<int> mNumberOfChipsPerStave;      ///< number of chips per stave
  std::vector<int> mNumberOfChipsPerHalfBarrel; ///< number of chips per halfbarrel
  std::vector<int> mNumberOfChipsPerLayer;      ///< number of chips per stave
  std::vector<int> mLastChipIndex;              ///< max ID of the detctor in the layer
  std::array<bool, MAXLAYERS> mIsLayerITS3;     ///< flag with the information of the ITS version (ITS2 or ITS3)
  std::array<char, MAXLAYERS> mLayerToWrapper;  ///< Layer to wrapper correspondence

  std::vector<float> mCacheRefX;     ///< sensors tracking plane reference X
  std::vector<float> mCacheRefAlpha; ///< sensors tracking plane reference alpha

  static std::string sVolumeName;        ///< Mother volume name
  static std::string sLayerName;         ///< Layer name
  static std::string sHalfBarrelName;    ///< HalfBarrel name
  static std::string sStaveName;         ///< Stave name
  static std::string sHalfStaveName;     ///< HalfStave name
  static std::string sModuleName;        ///< Module name
  static std::string sChipName;          ///< Chip name
  static std::string sSensorName;        ///< Sensor name
  static std::string sWrapperVolumeName; ///< Wrapper volume name

  static std::string sLayerNameITS3;      ///< Layer name for ITS3
  static std::string sHalfBarrelNameITS3; ///< HalfBarrel name for ITS3
  static std::string sStaveNameITS3;      ///< Stave name for ITS3
  static std::string sHalfStaveNameITS3;  ///< HalfStave name for ITS3
  static std::string sModuleNameITS3;     ///< Module name for ITS3
  static std::string sChipNameITS3;       ///< Chip name for ITS3
  static std::string sSensorNameITS3;     ///< Sensor name for ITS3

 private:
#ifndef GPUCA_STANDALONE
  static std::unique_ptr<o2::its::GeometryTGeo> sInstance; ///< singletone instance
#endif

  ClassDefOverride(GeometryTGeo, 1); // ITS geometry based on TGeo
};
} // namespace its
} // namespace o2

#endif
