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

namespace o2 {
namespace ITS {

/// Helper class to extract information from the GeoManager which is needed in many other TOF
/// classes. This helper class should be a single place to hold all these functions.
class GeometryHandler : public TObject
{

  public:
    /// Default constructor
    GeometryHandler();

    /// Default destructor
    ~GeometryHandler()
    override = default;

    Int_t getUniqueDetectorId();

    Int_t getUniqueDetectorId(TString volumeName);

    //  Int_t GetDetectorId(Int_t uniqueId);

    Int_t Init(Bool_t isSimulation = kFALSE);

    void fillDetectorInfoArray(Int_t uniqueId);

    void navigateTo(TString volumeName);

    // Implement Interface functions to the TGeoManager to be
    // the same as for the VMC

    /// Return the current volume "off" upward in the geometrical tree ID and copy number
    Int_t currentVolumeOffId(Int_t off, Int_t &copy) const;

    /// Returns the current volume ID and copy number
    Int_t currentVolumeId(Int_t &copy) const;

    /// Returns the unique numeric identifier for volume name
    Int_t volumeId(const Text_t *name) const;

    /// Returns the unique numeric identifier for volume name
    Int_t volumeIdGeo(const char *name) const;

    /// Returns the current volume name
    const char *currentVolumeName() const;

    /// Returns the current volume "off" upward in the geometrical tree ID, name and copy number
    /// if name=0 no name is returned
    const char *currentVolumeOffName(Int_t off) const;

    void localToGlobal(Double_t *local, Double_t *global, Int_t detectorId);

    //  Int_t CheckGeometryVersion();

  private:
    Bool_t mIsSimulation; //!

    Int_t mLastUsedDetectorId; //!

    UInt_t mGeometryPathHash;   //!
    TGeoVolume *mCurrentVolume; //!
    TGeoBBox *mVolumeShape;     //!
    Double_t mGlobalCentre[3];  //! Global centre of volume
    TGeoHMatrix *mGlobalMatrix; //!

    TString constructFullPathFromDetectorId(Int_t detectorId);

    GeometryHandler(const GeometryHandler &);

    GeometryHandler operator=(const GeometryHandler &);

  ClassDefOverride(GeometryHandler, 1)
};
}
}

#endif
