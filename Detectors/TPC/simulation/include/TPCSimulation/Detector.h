
#ifndef AliceO2_TPC_Detector_H_
#define AliceO2_TPC_Detector_H_

#include "DetectorsBase/Detector.h"   // for Detector
#include "Rtypes.h"          // for Int_t, Double32_t, Double_t, Bool_t, etc
#include "TLorentzVector.h"  // for TLorentzVector
#include "TVector3.h"        // for TVector3
#include "TString.h"

class FairVolume;  // lines 10-10
class TClonesArray;  // lines 11-11
namespace AliceO2 { namespace TPC { class Point; } }  // lines 15-15

class AliTPCParam;

namespace AliceO2 {
namespace TPC {
class Point;

class Detector: public AliceO2::Base::Detector {

  public:

    /**      Name :  Detector Name
     *       Active: kTRUE for active detectors (ProcessHits() will be called)
     *               kFALSE for inactive detectors
    */
    Detector(const char* Name, Bool_t Active);

    /**      default constructor    */
    Detector();

    /**       destructor     */
    virtual ~Detector();

    /**      Initialization of the detector is done here    */
    virtual void   Initialize();

    /**       this method is called for each step during simulation
     *       (see FairMCApplication::Stepping())
    */
//     virtual Bool_t ProcessHitsOrig( FairVolume* v=0);
    virtual Bool_t ProcessHits( FairVolume* v=0);

    /**       Registers the produced collections in FAIRRootManager.     */
    virtual void   Register();

    /** Gets the produced collections */
    virtual TClonesArray* GetCollection(Int_t iColl) const ;

    /**      has to be called after each event to reset the containers      */
    virtual void   Reset();

    /**      Create the detector geometry        */
    void ConstructGeometry();

    /**      This method is an example of how to add your own point
     *       of type DetectorPoint to the clones array
    */
    Point* AddHit(Int_t trackID, Int_t detID,
                             TVector3 pos, TVector3 mom,
                             Double_t time, Double_t length,
                             Double_t eLoss);
    

    /// Copied from AliRoot - should go to someplace else
    
    /// Empirical ALEPH parameterization of the Bethe-Bloch formula, normalized to 1 at the minimum.
    /// @param bg Beta*Gamma of the incident particle
    /// @param kp* Parameters for the ALICE TPC
    /// @return Bethe-Bloch value in MIP units
    Double_t BetheBlochAleph(Double_t bg, Double_t kp1, Double_t kp2, Double_t kp3, Double_t kp4, Double_t kp5);

    
    /** The following methods can be implemented if you need to make
     *  any optional action in your detector during the transport.
    */

    virtual void   CopyClones( TClonesArray* cl1,  TClonesArray* cl2 ,
                               Int_t offset) {;}
    virtual void   SetSpecialPhysicsCuts();// {;}
    virtual void   EndOfEvent();
    virtual void   FinishPrimary() {;}
    virtual void   FinishRun() {;}
    virtual void   BeginPrimary() {;}
    virtual void   PostTrack() {;}
    virtual void   PreTrack() {;}
    virtual void   BeginEvent() {;}

    void SetGeoFileName(const TString file) { mGeoFileName=file;   }
    const TString& GetGeoFileName() const   { return mGeoFileName; }

  private:

    /** Track information to be stored until the track leaves the
    active volume.
    */
    Int_t          mTrackNumberID;           //!  track index
    Int_t          mVolumeID;          //!  volume id
    TLorentzVector mPosition;               //!  position at entrance
    TLorentzVector mMomentum;               //!  momentum at entrance
    Double32_t     mTime;              //!  time
    Double32_t     mLength;            //!  length
    Double32_t     mEnergyLoss;             //!  energy loss

    /// Create the detector materials
    virtual void CreateMaterials();
    /// Geant settings hack
    void GeantHack();

    /// Construct the detector geometry
    void LoadGeometryFromFile();
    /// Construct the detector geometry
    void ConstructTPCGeometry();

    /** Define the sensitive volumes of the geometry */
    void DefineSensitiveVolumes();

    /** container for data points */
    TClonesArray*  mPointCollection;

    TString mGeoFileName;                  /// Name of the file containing the TPC geometry

    Detector(const Detector&);
    Detector& operator=(const Detector&);

    ClassDef(Detector,1)
};
}
}

#endif // AliceO2_TPC_Detector_H_
