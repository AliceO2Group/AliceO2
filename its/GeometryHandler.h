#ifndef ALICEO2_ITS_GEOMETRYHANDLER_H_
#define ALICEO2_ITS_GEOMETRYHANDLER_H_

// -------------------------------------------------------------------------
// -----                 GeometryHandler header file                  -----
// -----                 Created 20/11/12  by F. Uhlig                 -----
// -------------------------------------------------------------------------

/** GeometryHandler.h
 ** Helper class to extract information from the GeoManager which is
 ** needed in many other TOF classes. This helper class should be a
 ** single place to hold all these functions.
 ** @author F. Uhlig <f.uhlig@gsi.de>
 **/

#include "TObject.h"                    // for TObject

#include "Rtypes.h"                     // for Int_t, Double_t, Bool_t, etc
#include "TString.h"                    // for TString

class TGeoBBox;
class TGeoVolume;
class TGeoHMatrix;

namespace AliceO2 {
namespace ITS {

class GeometryHandler : public TObject
{
  public:

    /** Constructor **/
    GeometryHandler();

    /** Destructor **/
    ~GeometryHandler() {};

    Int_t GetUniqueDetectorId();
    Int_t GetUniqueDetectorId(TString volName);

//  Int_t GetDetectorId(Int_t uniqueId);

    Int_t Init(Bool_t isSimulation=kFALSE);

    void FillDetectorInfoArray(Int_t uniqueId);
    void NavigateTo(TString volName);

    // Implement Interface functions to the TGeoManager to be
    // the same as for the VMC
    Int_t CurrentVolOffID(Int_t off, Int_t& copy) const;
    Int_t CurrentVolID(Int_t& copy) const;
    Int_t VolId(const Text_t* name) const;
    Int_t VolIdGeo(const char* name) const;
    const char* CurrentVolName() const;
    const char* CurrentVolOffName(Int_t off) const;

    void LocalToGlobal(Double_t* local, Double_t* global, Int_t detID);

//  Int_t CheckGeometryVersion();

  private:

    Bool_t fIsSimulation; //!

    Int_t fLastUsedDetectorID;  //!

    UInt_t fGeoPathHash;        //!
    TGeoVolume* fCurrentVolume; //!
    TGeoBBox* fVolumeShape;     //!
    Double_t fGlobal[3];        //! Global centre of volume
    TGeoHMatrix* fGlobalMatrix; //!


    TString ConstructFullPathFromDetID(Int_t detID);

    GeometryHandler(const GeometryHandler&);
    GeometryHandler operator=(const GeometryHandler&);

    ClassDef(GeometryHandler,1)

};
}
}

#endif
