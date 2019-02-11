#ifndef ALIMAGF_H
#define ALIMAGF_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//
// Interface between the TVirtualMagField and AliMagWrapCheb: wrapper to
// the set of magnetic field data + Tosca parameterization by 
// Chebyshev polynomials
// 
// Author: ruben.shahoyan@cern.ch
//

//#include <TGeoGlobalMagField.h>
#include <TVirtualMagField.h>

class AliMagFast;
class AliMagWrapCheb;

class AliMagF : public TVirtualMagField
{
 public:
  enum BMap_t      {k2kG, k5kG, k5kGUniform};
  enum BeamType_t  {kNoBeamField, kBeamTypepp, kBeamTypeAA, kBeamTypepA, kBeamTypeAp};
  enum PolarityConvention_t {kConvLHC,kConvDCS2008,kConvMap2005};
  enum             {kOverrideGRP=BIT(14)}; // don't recreate from GRP if set
  //
  AliMagF();
  AliMagF(const char *name, const char* title,Double_t factorSol=1., Double_t factorDip=1., 
	  BMap_t maptype = k5kG, BeamType_t btype=kBeamTypepp, Double_t benergy=-1, float a2z=1.0,
	  Int_t integ=2, Double_t fmax=15,const char* path="$(ALICE_ROOT)/data/maps/mfchebKGI_sym.root");
  AliMagF(const char *name, const char* title,Double_t factorSol, Double_t factorDip, 
	  BMap_t maptype, BeamType_t btype, Double_t benergy, Int_t integ, Double_t fmax,const char* path="$(ALICE_ROOT)/data/maps/mfchebKGI_sym.root");
  AliMagF(const AliMagF& src);             
  AliMagF& operator=(const AliMagF& src);
  virtual ~AliMagF();
  //
  virtual void Field(const Double_t *x, Double_t *b);
  void       GetTPCInt(const Double_t *xyz, Double_t *b)         const;
  void       GetTPCRatInt(const Double_t *xyz, Double_t *b)      const;
  void       GetTPCIntCyl(const Double_t *rphiz, Double_t *b)    const;
  void       GetTPCRatIntCyl(const Double_t *rphiz, Double_t *b) const;
  Double_t   GetBz(const Double_t *xyz)                          const;
  //
  void        AllowFastField(Bool_t v=kTRUE);
  AliMagFast* GetFastField()                                    const {return fFastField;}
  AliMagWrapCheb* GetMeasuredMap()                              const {return fMeasuredMap;}
  //
  // former AliMagF methods or their aliases
  void         SetFactorSol(Float_t fc=1.);
  void         SetFactorDip(Float_t fc=1.);
  Double_t     GetFactorSol()                                   const;
  Double_t     GetFactorDip()                                   const;
  Double_t     Factor()                                         const {return GetFactorSol();}
  Double_t     GetCurrentSol()                                  const {return GetFactorSol()*(fMapType==k2kG ? 12000:30000);}
  Double_t     GetCurrentDip()                                  const {return GetFactorDip()*6000;}
  Bool_t       IsUniform()                                      const {return fMapType == k5kGUniform;}
  //
  void         MachineField(const Double_t  *x, Double_t *b)    const;
  BMap_t       GetMapType()                                     const {return fMapType;}
  BeamType_t   GetBeamType()                                    const {return fBeamType;}
  const char*  GetBeamTypeText()                                const;
  Double_t     GetBeamEnergy()                                  const {return fBeamEnergy;}
  Double_t     Max()                                            const {return fMax;}
  Int_t        Integ()                                          const {return fInteg;}
  Int_t        PrecInteg()                                      const {return fPrecInteg;}  
  Double_t     SolenoidField()                                  const {return fFactorSol*fSolenoid;}
  //
  Char_t*      GetDataFileName()                                const {return (Char_t*)fParNames.GetName();}
  Char_t*      GetParamName()                                   const {return (Char_t*)fParNames.GetTitle();}
  void         SetDataFileName(const Char_t* nm)                      {fParNames.SetName(nm);}
  void         SetParamName(const Char_t* nm)                         {fParNames.SetTitle(nm);}
  virtual void Print(Option_t *opt)                             const;
  //
  Bool_t       LoadParameterization();
  static Int_t GetPolarityConvention()                                {return Int_t(fgkPolarityConvention);}
  static AliMagF* CreateFieldMap(Float_t l3Current=-30000., Float_t diCurrent=-6000., 
				 Int_t convention=0, Bool_t uniform = kFALSE, 
				 Float_t beamenergy=7000, const Char_t* btype="pp", int az0=0, int az1=0,
				 const Char_t* path="$(ALICE_ROOT)/data/maps/mfchebKGI_sym.root",
				 Bool_t returnNullOnInvalidCurrent = kFALSE);
  //
  static void   SetFastFieldDefault(Bool_t v) {fgAllowFastField = v;}
  static Bool_t GetFastFieldDefault()         {return fgAllowFastField;}
  
 protected:
  // not supposed to be changed during the run, set only at the initialization via constructor
  void         InitMachineField(BeamType_t btype, Double_t benergy, float a2z=1.0);
  void         SetBeamType(BeamType_t type)                           {fBeamType = type;}
  void         SetBeamEnergy(Float_t energy)                          {fBeamEnergy = energy;}
  //
 protected:
  AliMagWrapCheb*  fMeasuredMap;     //! Measured part of the field map
  AliMagFast*      fFastField;       //! optional fast param
  BMap_t           fMapType;         // field map type
  Double_t         fSolenoid;        // Solenoid field setting
  BeamType_t       fBeamType;        // Beam type: A-A (fBeamType=0) or p-p (fBeamType=1)
  Double_t         fBeamEnergy;      // Beam energy in GeV
  // 
  Int_t            fInteg;           // Default integration method as indicated in Geant
  Int_t            fPrecInteg;       // Alternative integration method, e.g. for higher precision
  Double_t         fFactorSol;       // Multiplicative factor for solenoid
  Double_t         fFactorDip;       // Multiplicative factor for dipole
  Double_t         fMax;             // Max Field as indicated in Geant
  Bool_t           fDipoleOFF;       // Dipole ON/OFF flag
  //
  Double_t         fQuadGradient;    // Gradient field for inner triplet quadrupoles
  Double_t         fDipoleField;     // Field value for D1 and D2 dipoles
  Double_t         fCCorrField;      // Side C 2nd compensator field
  Double_t         fACorr1Field;     // Side A 1st compensator field 
  Double_t         fACorr2Field;     // Side A 2nd compensator field
  //
  TNamed           fParNames;        // file and parameterization loadad
  //
  static const Double_t  fgkSol2DipZ;    // conventional Z of transition from L3 to Dipole field
  static const UShort_t  fgkPolarityConvention; // convention for the mapping of the curr.sign on main component sign
  static Bool_t          fgAllowFastField;  // default setting for fast field usage
  //   
  ClassDef(AliMagF, 2)           // Class for all Alice MagField wrapper for measured data + Tosca parameterization
};


#endif
