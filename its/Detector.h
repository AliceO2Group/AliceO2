#ifndef ALICEO2_ITS_DETECTOR_H_
#define ALICEO2_ITS_DETECTOR_H_

#include "TParticle.h"
#include "TVector3.h"
#include "TLorentzVector.h"

#include "O2Detector.h"
#include "GeometryHandler.h"
#include "MisalignmentParameter.h"

#include "UpgradeGeometryTGeo.h"

class FairVolume;
class TClonesArray;

namespace AliceO2 {
namespace ITS {

class Point;
class UpgradeV1Layer;

class Detector: public O2Detector
{

  public:
  
    typedef enum {
      kIBModelDummy=0,
      kIBModel0=1,
      kIBModel1=2, 
      kIBModel21=3,
      kIBModel22=4,
      kIBModel3=5,
      kOBModelDummy=6,
      kOBModel0=7,
      kOBModel1=8 
    } AliITSUModel_t;

    /**      Name :  Detector Name
     *       Active: kTRUE for active detectors (ProcessHits() will be called)
     *               kFALSE for inactive detectors
    */
    Detector(const char* Name, Bool_t Active, const Int_t nlay);

    /**      default constructor    */
    Detector();

    /**       destructor     */
    virtual ~Detector();
    
    /**      Initialization of the detector is done here    */
    virtual void   Initialize();

    /**       this method is called for each step during simulation
     *       (see FairMCApplication::Stepping())
    */
    virtual Bool_t ProcessHits( FairVolume* v=0);

    /**       Registers the produced collections in FAIRRootManager.     */
    virtual void   Register();

    /** Gets the produced collections */
    virtual TClonesArray* GetCollection(Int_t iColl) const ;

    /**      has to be called after each event to reset the containers      */
    virtual void   Reset();

    /** Base class to create the detector geometry */
    void ConstructGeometry();

		void CreateSuppCyl(const Bool_t innerBarrel,TGeoVolume *dest,const TGeoManager *mgr=gGeoManager);
    
    /**      Initialize the parameter containers    */
    virtual void   InitParContainers();
    
    void SetParContainers();
    
    virtual void   DefineLayer(Int_t nlay,Double_t phi0,Double_t r,Double_t zlen,Int_t nladd,
			     Int_t nmod, Double_t lthick=0.,Double_t dthick=0.,UInt_t detType=0, Int_t buildFlag=0);
    virtual void   DefineLayerTurbo(Int_t nlay,Double_t phi0,Double_t r,Double_t zlen,Int_t nladd,
				  Int_t nmod,Double_t width,Double_t tilt,
				  Double_t lthick = 0.,Double_t dthick = 0.,UInt_t detType=0, Int_t buildFlag=0);
    virtual void   GetLayerParameters(Int_t nlay, Double_t &phi0,Double_t &r, Double_t &zlen,
				    Int_t &nladd, Int_t &nmod,
				    Double_t &width, Double_t &tilt,
				    Double_t &lthick, Double_t &mthick,
				    UInt_t &dettype) const;

    /**      This method is an example of how to add your own point
     *       of type Point to the clones array
    */
    Point* AddHit(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom,
                             Double_t startTime, Double_t time, Double_t length, Double_t eLoss, 
                             Int_t shunt);
    
    /** Book arrays for wrapper volumes */                         
    virtual void   SetNWrapVolumes(Int_t n);
    
    /** Set per wrapper volume parameters */
    virtual void   DefineWrapVolume(Int_t id, Double_t rmin,Double_t rmax, Double_t zspan);

    /** The following methods can be implemented if you need to make
     *  any optional action in your detector during the transport.
    */

    virtual void   CopyClones( TClonesArray* cl1,  TClonesArray* cl2 ,
                               Int_t offset) {;}
    virtual void   SetSpecialPhysicsCuts() {;}
    virtual void   EndOfEvent();
    virtual void   FinishPrimary() {;}
    virtual void   FinishRun() {;}
    virtual void   BeginPrimary() {;}
    virtual void   PostTrack() {;}
    virtual void   PreTrack() {;}
    virtual void   BeginEvent() {;}
    
    /** Returns a ptr to this particle. */
    virtual TParticle * GetParticle() const;
    
    /** SetTrack and GetTrack methods From AliHit.h */
    virtual void SetTrack(Int_t track) {fTrack=track;}
    virtual Int_t GetTrack() const {return fTrack;}
    
    /** Prints out the content of this class in ASCII format */
    void Print(ostream *os) const;
    
    /** Reads in the content of this class in the format of Print */
    void Read(istream *is);
    
    /** Returns the number of layers */
    Int_t   GetNLayers()              const {return fNLayers;}
    
    virtual void   SetStaveModelIB(AliITSUModel_t model) {fStaveModelIB=model;}
    virtual void   SetStaveModelOB(AliITSUModel_t model) {fStaveModelOB=model;}
    virtual AliITSUModel_t GetStaveModelIB() const {return fStaveModelIB;}
    virtual AliITSUModel_t GetStaveModelOB() const {return fStaveModelOB;}

    UpgradeGeometryTGeo* fGeomTGeo; //! access to geometry details
    
  protected:
  
    Int_t     *fIdSens;   //! [fNLayers] layer identifier
    Int_t     fNLayers;   //! Number of layers
    Int_t     fStatus;    //! Track Status
    Int_t     fModule;    //! Module number 
    Float_t   fPx;        //! PX of particle at the point of the hit
    Float_t   fPy;        //! PY of particle at the point of the hit
    Float_t   fPz;        //! PZ of particle at the point of the hit
    Float_t   fDestep;    //! Energy deposited in the current step
    Float_t   fTof;       //! Time of flight at the point of the hit
    Int_t     fStatus0;   //! Track Status of Starting point
    Float_t   fx0;        //! Starting point of this step
    Float_t   fy0;        //! Starting point of this step
    Float_t   fz0;        //! Starting point of this step
    Float_t   ft0;        //! Starting point of this step
    
    Int_t     fTrack;     //! Track number
    Float_t   fX;         //! X position of the hit
    Float_t   fY;         //! Y position of the hit
    Float_t   fZ;         //! Z position of the hit
    
    TString  *fLayerName; //![fNLayers] layer identifier
    
  private:

    /** Track information to be stored until the track leaves the
    active volume.
    */
    Int_t          fTrackID;           //!  track index
    Int_t          fVolumeID;          //!  volume id
    Int_t          fShunt;             //!  shunt
    TLorentzVector fPos;               //!  position
    TLorentzVector fStartPos;          //!  position at entrance
    TLorentzVector fMom;               //!  momentum
    Double32_t     fStartTime;         //!  time at entrance
    Double32_t     fTime;              //!  time
    Double32_t     fLength;            //!  length
    Double32_t     fELoss;             //!  energy loss
    
    Int_t fNrOfDetectors;
    TArrayD fShiftX;
    TArrayD fShiftY;
    TArrayD fShiftZ;
    TArrayD fRotX;
    TArrayD fRotY;
    TArrayD fRotZ;

    Bool_t fModifyGeometry;
    
    Int_t     fNWrapVol;       // number of wrapper volumes
  	Double_t* fWrapRMin;       // min radius of wrapper volume
	  Double_t* fWrapRMax;       // max radius of wrapper volume
  	Double_t* fWrapZSpan;      // Z span of wrapper volume
  	Int_t*    fLay2WrapV;      // id of wrapper layer to which layer belongs (-1 if not wrapped)
  	Bool_t   *fLayTurbo;       // True for "turbo" layers
  	Double_t *fLayPhi0;        // Vector of layer's 1st stave phi in lab
  	Double_t *fLayRadii;       // Vector of layer radii
  	Double_t *fLayZLength;     // Vector of layer length along Z
  	Int_t    *fStavPerLay;     // Vector of number of staves per layer
  	Int_t    *fUnitPerStave;   // Vector of number of "units" per stave
  	Double_t *fStaveThick;     // Vector of stave thicknesses
  	Double_t *fStaveWidth;     // Vector of stave width (only used for turbo)
  	Double_t *fStaveTilt;      // Vector of stave tilt (only used for turbo)
  	Double_t *fDetThick;       // Vector of detector thicknesses
  	UInt_t   *fChipTypeID;     // Vector of detector type id
  	Int_t    *fBuildLevel;     // Vector of Material Budget Studies

    /** Container for data points */
    TClonesArray*  fO2itsPointCollection;

    /** Creates an air-filled wrapper cylindrical volume */
    TGeoVolume* CreateWrapperVolume(const Int_t nLay);
    
    /** Create the detector materials */
    virtual void CreateMaterials();
    
    /** Construct the detector geometry */
    void ConstructDetectorGeometry();
    
    /** Define the sensitive volumes of the geometry */
    void DefineSensitiveVolumes();
    
    Detector(const Detector&);
    Detector& operator=(const Detector&);
    
    GeometryHandler* fGeoHandler;
    MisalignmentParameter* fMisalignPar;
    
    UpgradeV1Layer **fUpGeom; //! Geometry
    AliITSUModel_t fStaveModelIB; //! The stave model for the Inner Barrel
    AliITSUModel_t fStaveModelOB; //! The stave model for the Outer Barrel
    
    ClassDef(Detector,1)
};

// Input and output function for standard C++ input/output.
ostream& operator<<(ostream &os, Detector &source);
istream& operator>>(istream &os, Detector &source);

}
}

#endif
