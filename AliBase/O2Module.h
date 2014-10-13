#ifndef O2MODULE_H
#define O2MODULE_H

//
// This is the basic class for any
// ALICE detector module, whether it is 
// sensitive or not. Detector classes depend
// on this.
//

#include "FairModule.h"


class O2Module : public FairModule {

public:

  // Creators - distructors
  O2Module(const char* name, const char *title, Bool_t Active=kFALSE);
  O2Module();
  virtual ~O2Module();

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
  
protected:      

  static Float_t fgDensityFactor; //! factor that is multiplied to all material densities (ONLY for systematic studies)
  
 private:
 
  O2Module(const O2Module&);
  O2Module& operator=(const O2Module&);

  ClassDef(O2Module, 1)  //Base class for ALICE Modules
};
#endif
