/// \file GeometryTGeo.h
/// \brief Definition of the GeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012

#ifndef ALICEO2_ITS_GEOMETRYTGEO_H_
#define ALICEO2_ITS_GEOMETRYTGEO_H_

#include <TGeoMatrix.h> // for TGeoHMatrix
#include <TObjArray.h>  // for TObjArray
#include <TObject.h>    // for TObject
#include <TString.h>    // for TString
#include "Rtypes.h"     // for Int_t, Double_t, Bool_t, UInt_t, etc

class TGeoPNEntry; // lines 17-17

namespace o2
{
  namespace ITSMFT
  {
    class Segmentation;
  }
}

namespace o2
{
namespace ITS
{
// Adapted from the AliITSUAux class
const UInt_t gMaxLayers = 15; ///< max number of active layers

/// GeometryTGeo is a simple interface class to TGeoManager. It is used in the simulation
/// and reconstruction in order to query the TGeo ITS geometry.
/// RS: In order to preserve the static character of the class but make it dynamically access
/// geometry, we need to check in every method if the structures are initialized. To be converted
/// to singleton at later stage.
/// Note on the upgrade chip types:
/// The coarse type defines chips served by different classes, like Pix. Each such a chip type can
/// have kMaxSegmPerChipType segmentations (pitch etc.) whose parameteres are stored in the
/// Segmentation derived class (like SegmentationPixel). This allows to have in the setup
/// chips served by the same classes but with different segmentations. The full chip type is
/// composed as:
/// CoarseType*kMaxSegmPerChipType + segmentationType
/// The only requirement on the segmentationType that should be < kMaxSegmPerChipType.
/// The methods like getLayerChipTypeID return the full chip type
class GeometryTGeo : public TObject
{
 public:
  enum { kITSVNA, kITSVUpg }; // ITS version

  enum {
    kChipTypePix = 0,
    kNChipTypes,
    kMaxSegmPerChipType = 10
  }; // defined detector chip types (each one can have different segmentations)

  GeometryTGeo(Bool_t build = kFALSE, Bool_t loadSegmentationsentations = kTRUE);

  /// Default destructor
  ~GeometryTGeo() override;

  GeometryTGeo(const GeometryTGeo& src);

  /// Exract ITS parameters from TGeo
  void Build(Bool_t loadSegmentations);

  GeometryTGeo& operator=(const GeometryTGeo& geom);

  Int_t getNumberOfChips() const { return mNumberOfChips; }
  Int_t getNumberOfChipRowsPerModule(Int_t lay) const { return mNumberOfChipRowsPerModule[lay]; }
  Int_t getNumberOfChipColsPerModule(Int_t lay) const
  {
    return mNumberOfChipRowsPerModule[lay] ? mNumberOfChipsPerModule[lay] / mNumberOfChipRowsPerModule[lay] : -1;
  }

  Int_t getNumberOfChipsPerModule(Int_t lay) const { return mNumberOfChipsPerModule[lay]; }
  Int_t getNumberOfChipsPerHalfStave(Int_t lay) const { return mNumberOfChipsPerHalfStave[lay]; }
  Int_t getNumberOfChipsPerStave(Int_t lay) const { return mNumberOfChipsPerStave[lay]; }
  Int_t getNumberOfChipsPerLayer(Int_t lay) const { return mNumberOfChipsPerLayer[lay]; }
  Int_t getNumberOfModules(Int_t lay) const { return mNumberOfModules[lay]; }
  Int_t getNumberOfHalfStaves(Int_t lay) const { return mNumberOfHalfStaves[lay]; }
  Int_t getNumberOfStaves(Int_t lay) const { return mNumberOfStaves[lay]; }
  Int_t getNumberOfLayers() const { return mNumberOfLayers; }
  Int_t getChipIndex(Int_t lay, int detInLay) const { return getFirstChipIndex(lay) + detInLay; }
  /// This routine computes the chip index number from the layer, stave, and chip number in stave
  /// \param Int_t lay The layer number. Starting from 0.
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t chipInStave The chip number in the stave. Starting from 0
  Int_t getChipIndex(Int_t lay, Int_t sta, Int_t detInSta) const;

  /// This routine computes the chip index number from the layer, stave, substave and chip number
  /// in substave
  /// \param Int_t lay The layer number. Starting from 0.
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t substa The substave number. Starting from 0
  /// \param Int_t chipInSStave The chip number in the sub stave. Starting from 0
  Int_t getChipIndex(Int_t lay, Int_t sta, Int_t subSta, Int_t detInSubSta) const;

  /// This routine computes the chip index number from the layer,stave, substave module and
  /// chip number in module.
  /// \param Int_t lay The layer number. Starting from 0.
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t substa The substave number. Starting from 0
  /// \param Int_t module The module number ...
  /// \param Int_t chipInSStave The chip number in the module. Starting from 0
  Int_t getChipIndex(Int_t lay, Int_t sta, Int_t subSta, Int_t md, Int_t detInMod) const;

  /// This routine computes the layer, stave, substave, module and chip number
  /// given the chip index number
  /// \param Int_t index The chip index number, starting from zero.
  /// \param Int_t lay The layer number. Starting from 0
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t ssta The halfstave number. Starting from 0
  /// \param Int_t mod The module number. Starting from 0
  /// \param Int_t chip The detector number. Starting from 0
  Bool_t getChipId(Int_t index, Int_t& lay, Int_t& sta, Int_t& ssta, Int_t& mod, Int_t& chip) const;

  /// Get chip layer, from 0
  Int_t getLayer(Int_t index) const;

  /// Get chip stave, from 0
  Int_t getStave(Int_t index) const;

  /// Get chip substave id in stave, from 0
  Int_t getHalfStave(Int_t index) const;

  /// Get chip module id in substave, from 0
  Int_t getModule(Int_t index) const;

  /// Get chip number within layer, from 0
  Int_t getChipIdInLayer(Int_t index) const;

  /// Get chip number within stave, from 0
  Int_t getChipIdInStave(Int_t index) const;

  /// Get chip number within stave, from 0
  Int_t getChipIdInHalfStave(Int_t index) const;

  /// Get chip number within module, from 0
  Int_t getChipIdInModule(Int_t index) const;

  Int_t getLastChipIndex(Int_t lay) const { return mLastChipIndex[lay]; }
  Int_t getFirstChipIndex(Int_t lay) const { return (lay == 0) ? 0 : mLastChipIndex[lay - 1] + 1; }
  /// Get the TGeoPNEntry symbolic name for a given chip identified by 'index'
  const char* getSymbolicName(Int_t index) const;

  const char* getSymbolicName(Int_t lay, Int_t sta, Int_t det) const;

  // Attention: these are the matrices for the alignable volumes of the chips, i.e. not necessarily
  // the sensors

  /// Get the transformation matrix for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* GetMatrix(Int_t index) const;

  TGeoHMatrix* GetMatrix(Int_t lay, Int_t sta, Int_t det) const;

  /// Get the translation vector for a given chip 'index' by quering the TGeoManager
  Bool_t GetTranslation(Int_t index, Double_t t[3]) const;

  Bool_t GetTranslation(Int_t lay, Int_t sta, Int_t det, Double_t t[3]) const;

  /// Get the rotation matrix for a given chip 'index' by quering the TGeoManager
  Bool_t getRotation(Int_t index, Double_t r[9]) const;

  Bool_t getRotation(Int_t lay, Int_t sta, Int_t det, Double_t r[9]) const;

  /// Get the original (ideal geometry) TGeo matrix for a given chip identified by 'index'
  /// The method is slow, so it should be used with great care
  Bool_t GetOriginalMatrix(Int_t index, TGeoHMatrix& m) const;

  Bool_t GetOriginalMatrix(Int_t lay, Int_t sta, Int_t det, TGeoHMatrix& m) const;

  /// Get the original translation vector (ideal geometry)
  /// for a given chip 'index' by quering the TGeoManager
  Bool_t getOriginalTranslation(Int_t index, Double_t t[3]) const;

  Bool_t getOriginalTranslation(Int_t lay, Int_t sta, Int_t det, Double_t t[3]) const;

  /// Get the original rotation matrix (ideal geometry)
  /// for a given chip 'index' by quering the TGeoManager
  Bool_t getOriginalRotation(Int_t index, Double_t r[9]) const;

  Bool_t getOriginalRotation(Int_t lay, Int_t sta, Int_t det, Double_t r[9]) const;

  const TGeoHMatrix* getMatrixT2L(Int_t index);

  const TGeoHMatrix* getMatrixT2L(Int_t lay, Int_t sta, Int_t det) { return getMatrixT2L(getChipIndex(lay, sta, det)); }
  const TGeoHMatrix* getMatrixSensor(Int_t index);

  const TGeoHMatrix* getMatrixSensor(Int_t lay, Int_t sta, Int_t det)
  {
    return getMatrixSensor(getChipIndex(lay, sta, det));
  }

  /// Get the matrix which transforms from the tracking r.s. to the global one
  /// Returns kFALSE in case of error.
  Bool_t getTrackingMatrix(Int_t index, TGeoHMatrix& m);

  Bool_t getTrackingMatrix(Int_t lay, Int_t sta, Int_t det, TGeoHMatrix& m);

  // Attention: these are transformations wrt sensitive volume!
  void localToGlobal(Int_t index, const Double_t* loc, Double_t* glob);

  void localToGlobal(Int_t lay, Int_t sta, Int_t det, const Double_t* loc, Double_t* glob);

  void globalToLocal(Int_t index, const Double_t* glob, Double_t* loc);

  void globalToLocal(Int_t lay, Int_t sta, Int_t det, const Double_t* glob, Double_t* loc);

  void localToGlobalVector(Int_t index, const Double_t* loc, Double_t* glob);

  void globalToLocalVector(Int_t index, const Double_t* glob, Double_t* loc);

  Int_t getLayerChipTypeId(Int_t lr) const;

  Int_t getChipChipTypeId(Int_t id) const;

  const o2::ITSMFT::Segmentation* getSegmentationById(Int_t id) const;

  const o2::ITSMFT::Segmentation* getSegmentation(Int_t lr) const;

  TObjArray* getSegmentations() const { return (TObjArray*)mSegmentations; }
  void Print(Option_t* opt = "") const override;

  static UInt_t getUIDShift() { return mUIDShift; }
  static void setUIDShift(UInt_t s = 16) { mUIDShift = s < 16 ? s : 16; }
  static const char* getITSVolPattern() { return mVolumeName.Data(); }
  static const char* getITSLayerPattern() { return mLayerName.Data(); }
  static const char* getITSWrapVolPattern() { return mWrapperVolumeName.Data(); }
  static const char* getITSStavePattern() { return mStaveName.Data(); }
  static const char* getITSHalfStavePattern() { return mHalfStaveName.Data(); }
  static const char* getITSModulePattern() { return mModuleName.Data(); }
  static const char* getITSChipPattern() { return mChipName.Data(); }
  static const char* getITSSensorPattern() { return mSensorName.Data(); }
  static const char* getITSsegmentationFileName() { return mSegmentationFileName.Data(); }
  static const char* getChipTypeName(Int_t i);

  static void setITSVolPattern(const char* nm) { mVolumeName = nm; }
  static void setITSLayerPattern(const char* nm) { mLayerName = nm; }
  static void setITSWrapVolPattern(const char* nm) { mWrapperVolumeName = nm; }
  static void setITSStavePattern(const char* nm) { mStaveName = nm; }
  static void setITSHalfStavePattern(const char* nm) { mHalfStaveName = nm; }
  static void setITSModulePattern(const char* nm) { mModuleName = nm; }
  static void setITSChipPattern(const char* nm) { mChipName = nm; }
  static void setITSSensorPattern(const char* nm) { mSensorName = nm; }
  static void setChipTypeName(Int_t i, const char* nm);

  static void setITSsegmentationFileName(const char* nm) { mSegmentationFileName = nm; }
  static UInt_t composeChipTypeId(UInt_t segmId);

  /// sym name of the layer
  static const char* composeSymNameITS();

  /// sym name of the layer
  static const char* composeSymNameLayer(Int_t lr);

  /// Sym name of the stave at given layer
  static const char* composeSymNameStave(Int_t lr, Int_t sta);

  /// Sym name of the stave at given layer
  static const char* composeSymNameHalfStave(Int_t lr, Int_t sta, Int_t ssta);

  /// Sym name of the substave at given layer/stave
  static const char* composeSymNameModule(Int_t lr, Int_t sta, Int_t ssta, Int_t mod);

  /// Sym name of the chip in the given layer/stave/substave/module
  static const char* composeSymNameChip(Int_t lr, Int_t sta, Int_t ssta, Int_t mod, Int_t chip);

  // hack to avoid using AliGeomManager
  Int_t layerToVolUID(Int_t lay, int detInLay) const { return chipVolUID(getChipIndex(lay, detInLay)); }
  static Int_t chipVolUID(Int_t mod) { return (mod & 0xffff) << mUIDShift; }
 protected:
  /// Store pointer on often used matrices for faster access
  void fetchMatrices();

  void createT2LMatrices();

  /// Get the matrix which transforms from the tracking to local r.s.
  /// The method queries directly the TGeoPNEntry
  TGeoHMatrix* extractMatrixTrackingToLocal(Int_t index) const;

  /// Get the transformation matrix of the SENSOR (not necessary the same as the chip)
  /// for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* extractMatrixSensor(Int_t index) const;

  /// This routine computes the layer number a given the chip index
  /// \param Int_t index The chip index number, starting from zero.
  /// \param Int_t indexInLr The chip index inside a layer, starting from zero.
  /// \param Int_t lay The layer number. Starting from 0.
  Bool_t getLayer(Int_t index, Int_t& lay, Int_t& index2) const;

  /// Get a pointer to the TGeoPNEntry of a chip identified by 'index'
  /// Returns NULL in case of invalid index, missing TGeoManager or invalid symbolic name
  TGeoPNEntry* getPNEntry(Int_t index) const;

  /// Determines the number of chips per module on the (sub)stave in the Geometry
  /// Also extract the layout: span of module centers in Z and X
  /// \param lay: layer number from 0
  Int_t extractNumberOfChipsPerModule(Int_t lay, Int_t& nrow) const;

  /// Determines the number of layers in the Geometry
  /// \param lay: layer number, starting from 0
  Int_t extractNumberOfStaves(Int_t lay) const;

  /// Determines the number of substaves in the stave of the layer
  /// \param lay: layer number, starting from 0
  Int_t extractNumberOfHalfStaves(Int_t lay) const;

  /// Determines the number of modules in substave in the stave of the layer
  /// \param lay: layer number, starting from 0
  /// For the setup w/o modules defined the module and the stave or the substave is the same thing
  Int_t extractNumberOfModules(Int_t lay) const;

  /// Determines the layer detector type the Geometry and
  /// returns the detector type id for the layer
  /// \param lay: layer number from 0
  Int_t extractLayerChipType(Int_t lay) const;

  /// Determines the number of layers in the Geometry
  Int_t extractNumberOfLayers();

  /// Extract number following the prefix in the name string
  Int_t extractVolumeCopy(const char* name, const char* prefix) const;

 protected:
  Int_t mVersion;                 ///< ITS Version
  Int_t mNumberOfLayers;          ///< number of layers
  Int_t mNumberOfChips;           ///< The total number of chips
  Int_t* mNumberOfStaves;         //[mNumberOfLayers] Array of the number of staves/layer(layer)
  Int_t* mNumberOfHalfStaves;     //[mNumberOfLayers] Array of the number of substaves/stave(layer)
  Int_t* mNumberOfModules;        //[mNumberOfLayers] Array of the number of modules/substave(layer)
  Int_t* mNumberOfChipsPerModule; //[mNumberOfLayers] Array of the number of chips per module
  // (group of chips on the substaves)
  Int_t* mNumberOfChipRowsPerModule; //[mNumberOfLayers] Array of the number of chips rows per
  // module (relevant for OB modules)
  Int_t* mNumberOfChipsPerHalfStave;   //[mNumberOfLayers] Array of number of chips per substave
  Int_t* mNumberOfChipsPerStave;       //[mNumberOfLayers] Array of the number of chips per stave
  Int_t* mNumberOfChipsPerLayer;       //[mNumberOfLayers] Array of the number of chips per stave
  Int_t* mLayerChipType;               //[mNumberOfLayers] Array of layer chip types
  Int_t* mLastChipIndex;               //[mNumberOfLayers] max ID of the detctor in the layer
  Char_t mLayerToWrapper[gMaxLayers];  ///< Layer to wrapper correspondence
  TObjArray* mSensorMatrices;          ///< Sensor's matrices pointers in the geometry
  TObjArray* mTrackingToLocalMatrices; ///< Tracking to Local matrices pointers in the geometry
  TObjArray* mSegmentations;           ///< segmentations

  static UInt_t mUIDShift;                   ///< bit shift to go from mod.id to modUUID for TGeo
  static TString mVolumeName;                ///< Mother volume name
  static TString mLayerName;                 ///< Layer name
  static TString mStaveName;                 ///< Stave name
  static TString mHalfStaveName;             ///< HalfStave name
  static TString mModuleName;                ///< Module name
  static TString mChipName;                  ///< Chip name
  static TString mSensorName;                ///< Sensor name
  static TString mWrapperVolumeName;         ///< Wrapper volume name
  static TString mChipTypeName[kNChipTypes]; ///< upg detType Names

  static TString mSegmentationFileName; ///< file name for segmentations

  ClassDefOverride(GeometryTGeo, 1) // ITS geometry based on TGeo
};

/// Returns ymbolic name
inline const char* GeometryTGeo::getSymbolicName(Int_t lay, Int_t sta, Int_t det) const
{
  return getSymbolicName(getChipIndex(lay, sta, det));
}

/// Returns chip current matrix
inline TGeoHMatrix* GeometryTGeo::GetMatrix(Int_t lay, Int_t sta, Int_t det) const
{
  return GetMatrix(getChipIndex(lay, sta, det));
}

/// Returns translation
inline Bool_t GeometryTGeo::GetTranslation(Int_t lay, Int_t sta, Int_t det, Double_t t[3]) const
{
  return GetTranslation(getChipIndex(lay, sta, det), t);
}

/// Returns rotation
inline Bool_t GeometryTGeo::getRotation(Int_t lay, Int_t sta, Int_t det, Double_t r[9]) const
{
  return getRotation(getChipIndex(lay, sta, det), r);
}

/// Returns original matrix
inline Bool_t GeometryTGeo::GetOriginalMatrix(Int_t lay, Int_t sta, Int_t det, TGeoHMatrix& m) const
{
  return GetOriginalMatrix(getChipIndex(lay, sta, det), m);
}

/// Returns original translation
inline Bool_t GeometryTGeo::getOriginalTranslation(Int_t lay, Int_t sta, Int_t det, Double_t t[3]) const
{
  return getOriginalTranslation(getChipIndex(lay, sta, det), t);
}

/// Original rotation
inline Bool_t GeometryTGeo::getOriginalRotation(Int_t lay, Int_t sta, Int_t det, Double_t r[9]) const
{
  return getOriginalRotation(getChipIndex(lay, sta, det), r);
}

/// Tracking matrix
inline Bool_t GeometryTGeo::getTrackingMatrix(Int_t lay, Int_t sta, Int_t det, TGeoHMatrix& m)
{
  return getTrackingMatrix(getChipIndex(lay, sta, det), m);
}

/// Detector type ID of layer
inline Int_t GeometryTGeo::getLayerChipTypeId(Int_t lr) const { return mLayerChipType[lr]; }
// Detector type ID of chip
inline Int_t GeometryTGeo::getChipChipTypeId(Int_t id) const { return getLayerChipTypeId(getLayer(id)); }
/// Access global to sensor matrix
inline const TGeoHMatrix* GeometryTGeo::getMatrixSensor(Int_t index)
{
  if (!mSensorMatrices) {
    fetchMatrices();
  }
  return (TGeoHMatrix*)mSensorMatrices->At(index);
}

/// Access tracking to local matrix
inline const TGeoHMatrix* GeometryTGeo::getMatrixT2L(Int_t index)
{
  if (!mTrackingToLocalMatrices) {
    fetchMatrices();
  }
  return (TGeoHMatrix*)mTrackingToLocalMatrices->At(index);
}

/// Sensor local to global
inline void GeometryTGeo::localToGlobal(Int_t index, const Double_t* loc, Double_t* glob)
{
  getMatrixSensor(index)->LocalToMaster(loc, glob);
}

/// Global to sensor local
inline void GeometryTGeo::globalToLocal(Int_t index, const Double_t* glob, Double_t* loc)
{
  getMatrixSensor(index)->MasterToLocal(glob, loc);
}

/// Sensor local to global
inline void GeometryTGeo::localToGlobalVector(Int_t index, const Double_t* loc, Double_t* glob)
{
  getMatrixSensor(index)->LocalToMasterVect(loc, glob);
}

/// Global to sensor local
inline void GeometryTGeo::globalToLocalVector(Int_t index, const Double_t* glob, Double_t* loc)
{
  getMatrixSensor(index)->MasterToLocalVect(glob, loc);
}

/// Local2Master (sensor)
inline void GeometryTGeo::localToGlobal(Int_t lay, Int_t sta, Int_t det, const Double_t* loc, Double_t* glob)
{
  localToGlobal(getChipIndex(lay, sta, det), loc, glob);
}

/// Master2local (sensor)
inline void GeometryTGeo::globalToLocal(Int_t lay, Int_t sta, Int_t det, const Double_t* glob, Double_t* loc)
{
  globalToLocal(getChipIndex(lay, sta, det), glob, loc);
}

inline const char* GeometryTGeo::getChipTypeName(Int_t i)
{
  if (i >= kNChipTypes) {
    i /= kMaxSegmPerChipType; // full type is provided
  }
  return mChipTypeName[i].Data();
}

inline void GeometryTGeo::setChipTypeName(Int_t i, const char* nm)
{
  if (i >= kNChipTypes) {
    i /= kMaxSegmPerChipType; // full type is provided
  }
  mChipTypeName[i] = nm;
}

/// Get segmentation by ID
 inline const o2::ITSMFT::Segmentation* GeometryTGeo::getSegmentationById(Int_t id) const
{
  return mSegmentations ? (o2::ITSMFT::Segmentation*)mSegmentations->At(id) : nullptr;
}

/// Get segmentation of layer
 inline const o2::ITSMFT::Segmentation* GeometryTGeo::getSegmentation(Int_t lr) const
{
  return mSegmentations ? (o2::ITSMFT::Segmentation*)mSegmentations->At(getLayerChipTypeId(lr)) : nullptr;
}
}
}

#endif
