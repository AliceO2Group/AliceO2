/// \file GeometryHandler.h
/// \brief Definition of the GeometryHandler class
/// \author F. Uhlig <f.uhlig@gsi.de>

#ifndef ALICEO2_ITS_GEOMETRYHANDLER_H_
#define ALICEO2_ITS_GEOMETRYHANDLER_H_

#include "Rtypes.h"  // for Int_t, Double_t, Bool_t, etc
#include "TObject.h" // for TObject
#include "TString.h" // for TString

class TGeoBBox;
class TGeoVolume;
class TGeoHMatrix;

namespace AliceO2 {
namespace ITS {

/// Helper class to extract information from the GeoManager which is needed in many other TOF
/// classes. This helper class should be a single place to hold all these functions.
class GeometryHandler : public TObject {

public:
  /// Default constructor
  GeometryHandler();

  /// Default destructor
  ~GeometryHandler() {};

  Int_t GetUniqueDetectorId();

  Int_t GetUniqueDetectorId(TString volumeName);

  //  Int_t GetDetectorId(Int_t uniqueId);

  Int_t Init(Bool_t isSimulation = kFALSE);

  void FillDetectorInfoArray(Int_t uniqueId);

  void NavigateTo(TString volumeName);

  // Implement Interface functions to the TGeoManager to be
  // the same as for the VMC

  /// Return the current volume "off" upward in the geometrical tree ID and copy number
  Int_t CurrentVolumeOffId(Int_t off, Int_t& copy) const;

  /// Returns the current volume ID and copy number
  Int_t CurrentVolumeId(Int_t& copy) const;

  /// Returns the unique numeric identifier for volume name
  Int_t VolumeId(const Text_t* name) const;

  /// Returns the unique numeric identifier for volume name
  Int_t VolumeIdGeo(const char* name) const;

  /// Returns the current volume name
  const char* CurrentVolumeName() const;

  /// Returns the current volume "off" upward in the geometrical tree ID, name and copy number
  /// if name=0 no name is returned
  const char* CurrentVolumeOffName(Int_t off) const;

  void LocalToGlobal(Double_t* local, Double_t* global, Int_t detectorId);

  //  Int_t CheckGeometryVersion();

private:
  Bool_t mIsSimulation;

  Int_t mLastUsedDetectorId;

  UInt_t mGeometryPathHash;
  TGeoVolume* mCurrentVolume;
  TGeoBBox* mVolumeShape;
  Double_t mGlobalCentre[3]; ///< Global centre of volume
  TGeoHMatrix* mGlobalMatrix;

  TString ConstructFullPathFromDetectorId(Int_t detectorId);

  GeometryHandler(const GeometryHandler&);

  GeometryHandler operator=(const GeometryHandler&);

  ClassDef(GeometryHandler, 1)
};
}
}

#endif
