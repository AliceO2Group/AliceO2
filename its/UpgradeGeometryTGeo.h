/// \file UpgradeGeometryTGeo.h
/// \brief Definition of the UpgradeGeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012

#ifndef ALICEO2_ITS_UPGRADEGEOMETRYTGEO_H_
#define ALICEO2_ITS_UPGRADEGEOMETRYTGEO_H_

#include <TObject.h>
#include <TGeoMatrix.h>
#include <TString.h>
#include <TObjArray.h>

// FIXME: This is temporary and you have to remove it to avoid cyclic deps
#include <TGeoManager.h>

class TGeoPNEntry;
class TDatime;

namespace AliceO2 {
namespace ITS {

// Adapted from the AliITSUAux class
const UInt_t gMaxLayers = 15; ///< max number of active layers

class Segmentation;

/// UpgradeGeometryTGeo is a simple interface class to TGeoManager. It is used in the simulation
/// and reconstruction in order to query the TGeo ITS geometry.
/// RS: In order to preserve the static character of the class but make it dynamically access
/// geometry, we need to check in every method if the structures are initialized. To be converted
/// to singleton at later stage.
/// Note on the upgrade chip types:
/// The coarse type defines chips served by different classes, like Pix. Each such a chip type can
/// have kMaxSegmPerChipType segmentations (pitch etc.) whose parameteres are stored in the
/// Segmentation derived class (like UpgradeSegmentationPixel). This allows to have in the setup
/// chips served by the same classes but with different segmentations. The full chip type is
/// composed as:
/// CoarseType*kMaxSegmPerChipType + segmentationType
/// The only requirement on the segmentationType that should be < kMaxSegmPerChipType.
/// The methods like GetLayerChipTypeID return the full chip type
class UpgradeGeometryTGeo : public TObject {

public:
  enum { kITSVNA, kITSVUpg }; // ITS version

  enum {
    kChipTypePix = 0,
    kNChipTypes,
    kMaxSegmPerChipType = 10
  }; // defined detector chip types (each one can have different segmentations)

  UpgradeGeometryTGeo(Bool_t build = kFALSE, Bool_t loadSegmentationsentations = kTRUE);

  /// Default destructor
  virtual ~UpgradeGeometryTGeo();

  UpgradeGeometryTGeo(const UpgradeGeometryTGeo& src);

  UpgradeGeometryTGeo& operator=(const UpgradeGeometryTGeo& geom);

  Int_t GetNumberOfChips() const
  {
    return mNumberOfChips;
  }
  Int_t GetNumberOfChipRowsPerModule(Int_t lay) const
  {
    return mNumberOfChipRowsPerModule[lay];
  }
  Int_t GetNumberOfChipColsPerModule(Int_t lay) const
  {
    return mNumberOfChipRowsPerModule[lay]
             ? mNumberOfChipsPerModule[lay] / mNumberOfChipRowsPerModule[lay]
             : -1;
  }
  Int_t GetNumberOfChipsPerModule(Int_t lay) const
  {
    return mNumberOfChipsPerModule[lay];
  }
  Int_t GetNumberOfChipsPerHalfStave(Int_t lay) const
  {
    return mNumberOfChipsPerHalfStave[lay];
  }
  Int_t GetNumberOfChipsPerStave(Int_t lay) const
  {
    return mNumberOfChipsPerStave[lay];
  }
  Int_t GetNumberOfChipsPerLayer(Int_t lay) const
  {
    return mNumberOfChipsPerLayer[lay];
  }
  Int_t GetNumberOfModules(Int_t lay) const
  {
    return mNumberOfModules[lay];
  }
  Int_t GetNumberOfHalfStaves(Int_t lay) const
  {
    return mNumberOfHalfStaves[lay];
  }
  Int_t GetNumberOfStaves(Int_t lay) const
  {
    return mNumberOfStaves[lay];
  }
  Int_t GetNumberOfLayers() const
  {
    return mNumberOfLayers;
  }

  Int_t GetChipIndex(Int_t lay, int detInLay) const
  {
    return GetFirstChipIndex(lay) + detInLay;
  }

  /// This routine computes the chip index number from the layer, stave, and chip number in stave
  /// \param Int_t lay The layer number. Starting from 0.
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t chipInStave The chip number in the stave. Starting from 0
  Int_t GetChipIndex(Int_t lay, Int_t sta, Int_t detInSta) const;

  /// This routine computes the chip index number from the layer, stave, substave and chip number
  /// in substave
  /// \param Int_t lay The layer number. Starting from 0.
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t substa The substave number. Starting from 0
  /// \param Int_t chipInSStave The chip number in the sub stave. Starting from 0
  Int_t GetChipIndex(Int_t lay, Int_t sta, Int_t subSta, Int_t detInSubSta) const;

  /// This routine computes the chip index number from the layer,stave, substave module and
  /// chip number in module.
  /// \param Int_t lay The layer number. Starting from 0.
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t substa The substave number. Starting from 0
  /// \param Int_t module The module number ...
  /// \param Int_t chipInSStave The chip number in the module. Starting from 0
  Int_t GetChipIndex(Int_t lay, Int_t sta, Int_t subSta, Int_t md, Int_t detInMod) const;

  /// This routine computes the layer, stave, substave, module and chip number
  /// given the chip index number
  /// \param Int_t index The chip index number, starting from zero.
  /// \param Int_t lay The layer number. Starting from 0
  /// \param Int_t sta The stave number. Starting from 0
  /// \param Int_t ssta The halfstave number. Starting from 0
  /// \param Int_t mod The module number. Starting from 0
  /// \param Int_t chip The detector number. Starting from 0
  Bool_t GetChipId(Int_t index, Int_t& lay, Int_t& sta, Int_t& ssta, Int_t& mod, Int_t& chip) const;

  /// Get chip layer, from 0
  Int_t GetLayer(Int_t index) const;

  /// Get chip stave, from 0
  Int_t GetStave(Int_t index) const;

  /// Get chip substave id in stave, from 0
  Int_t GetHalfStave(Int_t index) const;

  /// Get chip module id in substave, from 0
  Int_t GetModule(Int_t index) const;

  /// Get chip number within layer, from 0
  Int_t GetChipIdInLayer(Int_t index) const;

  /// Get chip number within stave, from 0
  Int_t GetChipIdInStave(Int_t index) const;

  /// Get chip number within stave, from 0
  Int_t GetChipIdInHalfStave(Int_t index) const;

  /// Get chip number within module, from 0
  Int_t GetChipIdInModule(Int_t index) const;

  Int_t GetLastChipIndex(Int_t lay) const
  {
    return mLastChipIndex[lay];
  }
  Int_t GetFirstChipIndex(Int_t lay) const
  {
    return (lay == 0) ? 0 : mLastChipIndex[lay - 1] + 1;
  }

  /// Get the TGeoPNEntry symbolic name for a given chip identified by 'index'
  const char* GetSymbolicName(Int_t index) const;

  const char* GetSymbolicName(Int_t lay, Int_t sta, Int_t det) const;

  // Attention: these are the matrices for the alignable volumes of the chips, i.e. not necessarily
  // the sensors

  /// Get the transformation matrix for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* GetMatrix(Int_t index) const;
  TGeoHMatrix* GetMatrix(Int_t lay, Int_t sta, Int_t det) const;

  /// Get the translation vector for a given chip 'index' by quering the TGeoManager
  Bool_t GetTranslation(Int_t index, Double_t t[3]) const;
  Bool_t GetTranslation(Int_t lay, Int_t sta, Int_t det, Double_t t[3]) const;

  /// Get the rotation matrix for a given chip 'index' by quering the TGeoManager
  Bool_t GetRotation(Int_t index, Double_t r[9]) const;
  Bool_t GetRotation(Int_t lay, Int_t sta, Int_t det, Double_t r[9]) const;

  /// Get the original (ideal geometry) TGeo matrix for a given chip identified by 'index'
  /// The method is slow, so it should be used with great care
  Bool_t GetOriginalMatrix(Int_t index, TGeoHMatrix& m) const;
  Bool_t GetOriginalMatrix(Int_t lay, Int_t sta, Int_t det, TGeoHMatrix& m) const;

  /// Get the original translation vector (ideal geometry)
  /// for a given chip 'index' by quering the TGeoManager
  Bool_t GetOriginalTranslation(Int_t index, Double_t t[3]) const;
  Bool_t GetOriginalTranslation(Int_t lay, Int_t sta, Int_t det, Double_t t[3]) const;

  /// Get the original rotation matrix (ideal geometry)
  /// for a given chip 'index' by quering the TGeoManager
  Bool_t GetOriginalRotation(Int_t index, Double_t r[9]) const;
  Bool_t GetOriginalRotation(Int_t lay, Int_t sta, Int_t det, Double_t r[9]) const;

  const TGeoHMatrix* GetMatrixT2L(Int_t index);
  const TGeoHMatrix* GetMatrixT2L(Int_t lay, Int_t sta, Int_t det)
  {
    return GetMatrixT2L(GetChipIndex(lay, sta, det));
  }
  const TGeoHMatrix* GetMatrixSensor(Int_t index);
  const TGeoHMatrix* GetMatrixSensor(Int_t lay, Int_t sta, Int_t det)
  {
    return GetMatrixSensor(GetChipIndex(lay, sta, det));
  }

  /// Get the matrix which transforms from the tracking r.s. to the global one
  /// Returns kFALSE in case of error.
  Bool_t GetTrackingMatrix(Int_t index, TGeoHMatrix& m);
  Bool_t GetTrackingMatrix(Int_t lay, Int_t sta, Int_t det, TGeoHMatrix& m);

  // Attention: these are transformations wrt sensitive volume!
  void LocalToGlobal(Int_t index, const Double_t* loc, Double_t* glob);
  void LocalToGlobal(Int_t lay, Int_t sta, Int_t det, const Double_t* loc, Double_t* glob);

  void GlobalToLocal(Int_t index, const Double_t* glob, Double_t* loc);
  void GlobalToLocal(Int_t lay, Int_t sta, Int_t det, const Double_t* glob, Double_t* loc);

  void LocalToGlobalVector(Int_t index, const Double_t* loc, Double_t* glob);
  void GlobalToLocalVector(Int_t index, const Double_t* glob, Double_t* loc);

  Int_t GetLayerChipTypeId(Int_t lr) const;
  Int_t GetChipChipTypeId(Int_t id) const;

  const Segmentation* GetSegmentationById(Int_t id) const;
  const Segmentation* GetSegmentation(Int_t lr) const;

  TObjArray* GetSegmentations() const
  {
    return (TObjArray*)mSegmentations;
  }
  virtual void Print(Option_t* opt = "") const;

  static UInt_t GetUIDShift()
  {
    return mUIDShift;
  }
  static void SetUIDShift(UInt_t s = 16)
  {
    mUIDShift = s < 16 ? s : 16;
  }

  static const char* GetITSVolPattern()
  {
    return mVolumeName.Data();
  }
  static const char* GetITSLayerPattern()
  {
    return mLayerName.Data();
  }
  static const char* GetITSWrapVolPattern()
  {
    return mWrapperVolumeName.Data();
  }
  static const char* GetITSStavePattern()
  {
    return mStaveName.Data();
  }
  static const char* GetITSHalfStavePattern()
  {
    return mHalfStaveName.Data();
  }
  static const char* GetITSModulePattern()
  {
    return mModuleName.Data();
  }
  static const char* GetITSChipPattern()
  {
    return mChipName.Data();
  }
  static const char* GetITSSensorPattern()
  {
    return mSensorName.Data();
  }
  static const char* GetITSsegmentationFileName()
  {
    return mSegmentationFileName.Data();
  }
  static const char* GetChipTypeName(Int_t i);

  static void SetITSVolPattern(const char* nm)
  {
    mVolumeName = nm;
  }
  static void SetITSLayerPattern(const char* nm)
  {
    mLayerName = nm;
  }
  static void SetITSWrapVolPattern(const char* nm)
  {
    mWrapperVolumeName = nm;
  }
  static void SetITSStavePattern(const char* nm)
  {
    mStaveName = nm;
  }
  static void SetITSHalfStavePattern(const char* nm)
  {
    mHalfStaveName = nm;
  }
  static void SetITSModulePattern(const char* nm)
  {
    mModuleName = nm;
  }
  static void SetITSChipPattern(const char* nm)
  {
    mChipName = nm;
  }
  static void SetITSSensorPattern(const char* nm)
  {
    mSensorName = nm;
  }
  static void SetChipTypeName(Int_t i, const char* nm);
  static void SetITSsegmentationFileName(const char* nm)
  {
    mSegmentationFileName = nm;
  }
  static UInt_t ComposeChipTypeId(UInt_t segmId);

  /// sym name of the layer
  static const char* ComposeSymNameITS();

  /// sym name of the layer
  static const char* ComposeSymNameLayer(Int_t lr);

  /// Sym name of the stave at given layer
  static const char* ComposeSymNameStave(Int_t lr, Int_t sta);

  /// Sym name of the stave at given layer
  static const char* ComposeSymNameHalfStave(Int_t lr, Int_t sta, Int_t ssta);

  /// Sym name of the substave at given layer/stave
  static const char* ComposeSymNameModule(Int_t lr, Int_t sta, Int_t ssta, Int_t mod);

  /// Sym name of the chip in the given layer/stave/substave/module
  static const char* ComposeSymNameChip(Int_t lr, Int_t sta, Int_t ssta, Int_t mod, Int_t chip);

  // hack to avoid using AliGeomManager
  Int_t LayerToVolUID(Int_t lay, int detInLay) const
  {
    return ChipVolUID(GetChipIndex(lay, detInLay));
  }
  static Int_t ChipVolUID(Int_t mod)
  {
    return (mod & 0xffff) << mUIDShift;
  }

protected:
  /// Store pointer on often used matrices for faster access
  void FetchMatrices();
  void CreateT2LMatrices();

  /// Get the matrix which transforms from the tracking to local r.s.
  /// The method queries directly the TGeoPNEntry
  TGeoHMatrix* ExtractMatrixTrackingToLocal(Int_t index) const;

  /// Get the transformation matrix of the SENSOR (not necessary the same as the chip)
  /// for a given chip 'index' by quering the TGeoManager
  TGeoHMatrix* ExtractMatrixSensor(Int_t index) const;

  /// This routine computes the layer number a given the chip index
  /// \param Int_t index The chip index number, starting from zero.
  /// \param Int_t indexInLr The chip index inside a layer, starting from zero.
  /// \param Int_t lay The layer number. Starting from 0.
  Bool_t GetLayer(Int_t index, Int_t& lay, Int_t& index2) const;

  /// Get a pointer to the TGeoPNEntry of a chip identified by 'index'
  /// Returns NULL in case of invalid index, missing TGeoManager or invalid symbolic name
  TGeoPNEntry* GetPNEntry(Int_t index) const;

  /// Determines the number of chips per module on the (sub)stave in the Upgrade Geometry
  /// Also extract the layout: span of module centers in Z and X
  /// \param lay: layer number from 0
  Int_t ExtractNumberOfChipsPerModule(Int_t lay, Int_t& nrow) const;

  /// Determines the number of layers in the Upgrade Geometry
  /// \param lay: layer number, starting from 0
  Int_t ExtractNumberOfStaves(Int_t lay) const;

  /// Determines the number of substaves in the stave of the layer
  /// \param lay: layer number, starting from 0
  Int_t ExtractNumberOfHalfStaves(Int_t lay) const;

  /// Determines the number of modules in substave in the stave of the layer
  /// \param lay: layer number, starting from 0
  /// For the setup w/o modules defined the module and the stave or the substave is the same thing
  Int_t ExtractNumberOfModules(Int_t lay) const;

  /// Determines the layer detector type the Upgrade Geometry and
  /// returns the detector type id for the layer
  /// \param lay: layer number from 0
  Int_t ExtractLayerChipType(Int_t lay) const;

  /// Determines the number of layers in the Upgrade Geometry
  Int_t ExtractNumberOfLayers();

  /// Exract ITS Upgrade parameters from TGeo
  void Build(Bool_t loadSegmentations);

  /// Extract number following the prefix in the name string
  Int_t ExtractVolumeCopy(const char* name, const char* prefix) const;

protected:
  Int_t mVersion;             ///< ITS Version
  Int_t mNumberOfLayers;      ///< number of layers
  Int_t mNumberOfChips;       ///< The total number of chips
  Int_t* mNumberOfStaves;     ///< [mNumberOfLayers] Array of the number of staves/layer(layer)
  Int_t* mNumberOfHalfStaves; ///< [mNumberOfLayers] Array of the number of substaves/stave(layer)
  Int_t* mNumberOfModules;    ///< [mNumberOfLayers] Array of the number of modules/substave(layer)
  Int_t* mNumberOfChipsPerModule;    ///< [mNumberOfLayers] Array of the number of chips per module
                                     ///< (group of chips on the substaves)
  Int_t* mNumberOfChipRowsPerModule; ///< [mNumberOfLayers] Array of the number of chips rows per
  ///< module (relevant for OB modules)
  Int_t* mNumberOfChipsPerHalfStave;   ///< [mNumberOfLayers] Array of number of chips per substave
  Int_t* mNumberOfChipsPerStave;       ///< [mNumberOfLayers] Array of the number of chips per stave
  Int_t* mNumberOfChipsPerLayer;       ///< [mNumberOfLayers] Array of the number of chips per stave
  Int_t* mLayerChipType;               ///< [mNumberOfLayers] Array of layer chip types
  Int_t* mLastChipIndex;               ///< [mNumberOfLayers] max ID of the detctor in the layer
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

  ClassDef(UpgradeGeometryTGeo, 1) // ITS geometry based on TGeo
};

/// Returns ymbolic name
inline const char* UpgradeGeometryTGeo::GetSymbolicName(Int_t lay, Int_t sta, Int_t det) const
{
  return GetSymbolicName(GetChipIndex(lay, sta, det));
}

/// Returns chip current matrix
inline TGeoHMatrix* UpgradeGeometryTGeo::GetMatrix(Int_t lay, Int_t sta, Int_t det) const
{
  return GetMatrix(GetChipIndex(lay, sta, det));
}

/// Returns translation
inline Bool_t UpgradeGeometryTGeo::GetTranslation(Int_t lay, Int_t sta, Int_t det, Double_t t[3])
  const
{
  return GetTranslation(GetChipIndex(lay, sta, det), t);
}

/// Returns rotation
inline Bool_t UpgradeGeometryTGeo::GetRotation(Int_t lay, Int_t sta, Int_t det, Double_t r[9]) const
{
  return GetRotation(GetChipIndex(lay, sta, det), r);
}

/// Returns original matrix
inline Bool_t UpgradeGeometryTGeo::GetOriginalMatrix(Int_t lay, Int_t sta, Int_t det,
                                                     TGeoHMatrix& m) const
{
  return GetOriginalMatrix(GetChipIndex(lay, sta, det), m);
}

/// Returns original translation
inline Bool_t UpgradeGeometryTGeo::GetOriginalTranslation(Int_t lay, Int_t sta, Int_t det,
                                                          Double_t t[3]) const
{
  return GetOriginalTranslation(GetChipIndex(lay, sta, det), t);
}

/// Original rotation
inline Bool_t UpgradeGeometryTGeo::GetOriginalRotation(Int_t lay, Int_t sta, Int_t det,
                                                       Double_t r[9]) const
{
  return GetOriginalRotation(GetChipIndex(lay, sta, det), r);
}

/// Tracking matrix
inline Bool_t UpgradeGeometryTGeo::GetTrackingMatrix(Int_t lay, Int_t sta, Int_t det,
                                                     TGeoHMatrix& m)
{
  return GetTrackingMatrix(GetChipIndex(lay, sta, det), m);
}

/// Detector type ID of layer
inline Int_t UpgradeGeometryTGeo::GetLayerChipTypeId(Int_t lr) const
{
  return mLayerChipType[lr];
}

// Detector type ID of chip
inline Int_t UpgradeGeometryTGeo::GetChipChipTypeId(Int_t id) const
{
  return GetLayerChipTypeId(GetLayer(id));
}

/// Access global to sensor matrix
inline const TGeoHMatrix* UpgradeGeometryTGeo::GetMatrixSensor(Int_t index)
{
  if (!mSensorMatrices) {
    FetchMatrices();
  }
  return (TGeoHMatrix*)mSensorMatrices->At(index);
}

/// Access tracking to local matrix
inline const TGeoHMatrix* UpgradeGeometryTGeo::GetMatrixT2L(Int_t index)
{
  if (!mTrackingToLocalMatrices) {
    FetchMatrices();
  }
  return (TGeoHMatrix*)mTrackingToLocalMatrices->At(index);
}

/// Sensor local to global
inline void UpgradeGeometryTGeo::LocalToGlobal(Int_t index, const Double_t* loc, Double_t* glob)
{
  GetMatrixSensor(index)->LocalToMaster(loc, glob);
}

/// Global to sensor local
inline void UpgradeGeometryTGeo::GlobalToLocal(Int_t index, const Double_t* glob, Double_t* loc)
{
  GetMatrixSensor(index)->MasterToLocal(glob, loc);
}

/// Sensor local to global
inline void UpgradeGeometryTGeo::LocalToGlobalVector(Int_t index, const Double_t* loc,
                                                     Double_t* glob)
{
  GetMatrixSensor(index)->LocalToMasterVect(loc, glob);
}

/// Global to sensor local
inline void UpgradeGeometryTGeo::GlobalToLocalVector(Int_t index, const Double_t* glob,
                                                     Double_t* loc)
{
  GetMatrixSensor(index)->MasterToLocalVect(glob, loc);
}

/// Local2Master (sensor)
inline void UpgradeGeometryTGeo::LocalToGlobal(Int_t lay, Int_t sta, Int_t det, const Double_t* loc,
                                               Double_t* glob)
{
  LocalToGlobal(GetChipIndex(lay, sta, det), loc, glob);
}

/// Master2local (sensor)
inline void UpgradeGeometryTGeo::GlobalToLocal(Int_t lay, Int_t sta, Int_t det,
                                               const Double_t* glob, Double_t* loc)
{
  GlobalToLocal(GetChipIndex(lay, sta, det), glob, loc);
}

inline const char* UpgradeGeometryTGeo::GetChipTypeName(Int_t i)
{
  if (i >= kNChipTypes) {
    i /= kMaxSegmPerChipType; // full type is provided
  }
  return mChipTypeName[i].Data();
}

inline void UpgradeGeometryTGeo::SetChipTypeName(Int_t i, const char* nm)
{
  if (i >= kNChipTypes) {
    i /= kMaxSegmPerChipType; // full type is provided
  }
  mChipTypeName[i] = nm;
}

/// Get segmentation by ID
inline const Segmentation* UpgradeGeometryTGeo::GetSegmentationById(Int_t id) const
{
  return mSegmentations ? (Segmentation*)mSegmentations->At(id) : 0;
}

/// Get segmentation of layer
inline const Segmentation* UpgradeGeometryTGeo::GetSegmentation(Int_t lr) const
{
  return mSegmentations ? (Segmentation*)mSegmentations->At(GetLayerChipTypeId(lr)) : 0;
}
}
}

#endif
