
#ifndef AliceO2_TPC_Detector_H_
#define AliceO2_TPC_Detector_H_

#include "DetectorsBase/Detector.h"   // for Detector
#include "Rtypes.h"          // for Int_t, Double32_t, Double_t, Bool_t, etc
#include "TLorentzVector.h"  // for TLorentzVector
#include "TClonesArray.h"
#include "TString.h"

#include "TPCSimulation/Point.h"

class FairVolume;  // lines 10-10

class AliTPCParam;

namespace AliceO2 {
namespace TPC {

class Detector: public AliceO2::Base::Detector {

  public:
  enum class SimulationType : char {
    GEANT3,    ///< GEANT3 simulation
    Other      ///< Other simulation, e.g. GEANT4
      };


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
    virtual Bool_t ProcessHits( FairVolume* v=nullptr);

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
    Point* addHit(float x, float y, float z, float time, float nElectrons, float trackID, float detID);
    

    /// Copied from AliRoot - should go to someplace else
    
    /// Empirical ALEPH parameterization of the Bethe-Bloch formula, normalized to 1 at the minimum.
    /// @param bg Beta*Gamma of the incident particle
    /// @param kp* Parameters for the ALICE TPC
    /// @return Bethe-Bloch value in MIP units
    Double_t BetheBlochAleph(Double_t bg, Double_t kp1, Double_t kp2, Double_t kp3, Double_t kp4, Double_t kp5);

    /// Copied from AliRoot - should go to someplace else
    /// Function to generate random numbers according to Gamma function 
    /// From Hisashi Tanizaki:
    /// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.158.3866&rep=rep1&type=pdf
    /// Implemented by A. Morsch 14/01/2014    
    /// @k is the mean and variance
    Double_t Gamma(Double_t k);

    
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
    
    SimulationType mSimulationType;       ///< Type of simulation

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

inline
Point* Detector::addHit(float x, float y, float z, float time, float nElectrons, float trackID, float detID)
{
  TClonesArray& clref = *mPointCollection;
  Int_t size = clref.GetEntriesFast();
  return new(clref[size]) Point(x, y, z, time, nElectrons, trackID, detID);
}
}
}

#endif // AliceO2_TPC_Detector_H_
