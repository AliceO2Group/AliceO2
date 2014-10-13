#ifndef O2DETECTOR_H
#define O2DETECTOR_H

//
// This is the basic class for any
// ALICE detector module, whether it is 
// sensitive or not. Detector classes depend
// on this.
//

#include "FairDetector.h"

class O2Detector : public FairDetector {
public:

  // Creators - distructors
  O2Detector(const char* name, Bool_t Active, Int_t DetId=0);
  O2Detector();
  virtual ~O2Detector();

  // Module composition
  virtual void AliMaterial(Int_t imat, const char* name, Float_t a, 
			   Float_t z, Float_t dens, Float_t radl,
			   Float_t absl, Float_t *buf=0, Int_t nwbuf=0) const;
  virtual void AliMixture(Int_t imat, const char *name, Float_t *a,
                          Float_t *z, Float_t dens, Int_t nlmat,
                          Float_t *wmat) const;
  virtual void AliMedium(Int_t numed, const char *name, Int_t nmat,
                          Int_t isvol, Int_t ifield, Float_t fieldm,
                          Float_t tmaxfd, Float_t stemax, Float_t deemax,
                          Float_t epsil, Float_t stmin, Float_t *ubuf=0,
                          Int_t nbuf=0) const;
  virtual void AliMatrix(Int_t &nmat, Float_t theta1, Float_t phi1,
                          Float_t theta2, Float_t phi2, Float_t theta3,
                          Float_t phi3) const;
  
  static void SetDensityFactor(Float_t density) { fgDensityFactor = density; }
  static Float_t GetDensityFactor() { return fgDensityFactor; }
  
  /** Set per wrapper volume parameters */
  virtual void   DefineWrapVolume(Int_t id, Double_t rmin,Double_t rmax, Double_t zspan);

  /** Book arrays for wrapper volumes */                         
  virtual void   SetNWrapVolumes(Int_t n);
     
  virtual void   DefineLayer(Int_t nlay,Double_t phi0,Double_t r,Double_t zlen,Int_t nladd,
          Int_t nmod, Double_t lthick=0.,Double_t dthick=0.,UInt_t detType=0, Int_t buildFlag=0);

  virtual void   DefineLayerTurbo(Int_t nlay,Double_t phi0,Double_t r,Double_t zlen,Int_t nladd,
          Int_t nmod,Double_t width,Double_t tilt,
          Double_t lthick = 0.,Double_t dthick = 0.,UInt_t detType=0, Int_t buildFlag=0);
  
protected:      

  static Float_t fgDensityFactor; //! factor that is multiplied to all material densities (ONLY for systematic studies)
private:
  O2Detector(const O2Detector&);
  O2Detector& operator=(const O2Detector&);

ClassDef(O2Detector, 1)  //Base class for ALICE Modules
};
#endif
