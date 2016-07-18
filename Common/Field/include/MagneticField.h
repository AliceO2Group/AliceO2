/// \file MagF.h
/// \brief Definition of the MagF class
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGNETICFIELD_H_
#define ALICEO2_FIELD_MAGNETICFIELD_H_

#include <TVirtualMagField.h>  // for TVirtualMagField
#include "AliceO2Config.h"     // for O2PROTO1_MAGF_CREATEFIELDMAP_DIR, etc
#include "Rtypes.h"            // for Double_t, Char_t, Int_t, Float_t, etc
#include "TNamed.h"            // for TNamed
class FairLogger;  // lines 14-14
namespace AliceO2 { namespace Field { class MagneticWrapperChebyshev; } }  // lines 19-19
namespace AliceO2 {
namespace Field {

class MagneticWrapperChebyshev;

/// Interface between the TVirtualMagField and MagneticWrapperChebyshev: wrapper to the set of magnetic field data +
/// Tosca
/// parameterization by Chebyshev polynomials
class MagneticField : public TVirtualMagField {

public:
  enum BMap_t { k2kG, k5kG, k5kGUniform };
  enum BeamType_t { kNoBeamField, kBeamTypepp, kBeamTypeAA, kBeamTypepA, kBeamTypeAp };
  enum PolarityConvention_t { kConvLHC, kConvDCS2008, kConvMap2005 };
  enum { kOverrideGRP = BIT(14) }; // don't recreate from GRP if set

  /// Default constructor
  MagneticField();

  /// Initialize the field with Geant integration option "integ" and max field "fmax",
  /// Impose scaling of parameterized L3 field by factorSol and of dipole by factorDip.
  /// The "be" is the energy of the beam in GeV/nucleon
  MagneticField(const char* name, const char* title, Double_t factorSol = 1., Double_t factorDip = 1.,
                BMap_t maptype = k5kG, BeamType_t btype = kBeamTypepp, Double_t benergy = -1, Int_t integ = 2,
                Double_t fmax = 15, const char* path = O2PROTO1_MAGF_DIR);
  MagneticField(const MagneticField& src);
  MagneticField& operator=(const MagneticField& src);

  /// Default destructor
  virtual ~MagneticField();

  /// Method to calculate the field at point xyz
  virtual void Field(const Double_t* x, Double_t* b);

  /// Method to calculate the integral_0^z of br,bt,bz
  void getTPCIntegral(const Double_t* xyz, Double_t* b) const;

  /// Method to calculate the integral_0^z of br,bt,bz
  void getTPCRatIntegral(const Double_t* xyz, Double_t* b) const;

  /// Method to calculate the integral_0^z of br,bt,bz in cylindrical coordinates ( -pi<phi<pi convention )
  void getTPCIntegralCylindrical(const Double_t* rphiz, Double_t* b) const;

  /// Method to calculate the integral_0^z of bx/bz,by/bz and (bx/bz)^2+(by/bz)^2 in
  /// cylindrical coordiates ( -pi<phi<pi convention )
  void getTPCRatIntegralCylindrical(const Double_t* rphiz, Double_t* b) const;

  /// Method to calculate the field at point xyz
  Double_t getBz(const Double_t* xyz) const;

  MagneticWrapperChebyshev* getMeasuredMap() const
  {
    return mMeasuredMap;
  }

  // Former MagF methods or their aliases

  /// Sets the sign/scale of the current in the L3 according to sPolarityConvention
  void setFactorSolenoid(Float_t fc = 1.);

  /// Sets the sign*scale of the current in the Dipole according to sPolarityConvention
  void setFactorDipole(Float_t fc = 1.);

  /// Returns the sign*scale of the current in the Dipole according to sPolarityConventionthe
  Double_t getFactorSolenoid() const;

  /// Return the sign*scale of the current in the Dipole according to sPolarityConventionthe
  Double_t getFactorDipole() const;

  Double_t Factor() const
  {
    return getFactorSolenoid();
  }

  Double_t getCurrentSolenoid() const
  {
    return getFactorSolenoid() * (mMapType == k2kG ? 12000 : 30000);
  }

  Double_t getCurrentDipole() const
  {
    return getFactorDipole() * 6000;
  }

  Bool_t IsUniform() const
  {
    return mMapType == k5kGUniform;
  }

  void MachineField(const Double_t* x, Double_t* b) const;

  BMap_t getMapType() const
  {
    return mMapType;
  }

  BeamType_t getBeamType() const
  {
    return mBeamType;
  }

  /// Returns beam type in text form
  const char* getBeamTypeText() const;

  Double_t getBeamEnergy() const
  {
    return mBeamEnergy;
  }

  Double_t Max() const
  {
    return mMaxField;
  }

  Int_t Integral() const
  {
    return mDefaultIntegration;
  }

  Int_t precIntegral() const
  {
    return mPrecisionInteg;
  }

  Double_t solenoidField() const
  {
    return mMultipicativeFactorSolenoid * mSolenoid;
  }

  Char_t* getDataFileName() const
  {
    return (Char_t*)mParameterNames.GetName();
  }

  Char_t* getParameterName() const
  {
    return (Char_t*)mParameterNames.GetTitle();
  }

  void setDataFileName(const Char_t* nm)
  {
    mParameterNames.SetName(nm);
  }

  void setParameterName(const Char_t* nm)
  {
    mParameterNames.SetTitle(nm);
  }

  /// Prints short or long info
  virtual void Print(Option_t* opt) const;

  Bool_t loadParameterization();

  static Int_t getPolarityConvention()
  {
    return Int_t(sPolarityConvention);
  }

  /// The magnetic field map, defined externally...
  /// L3 current 30000 A  -> 0.5 T
  /// L3 current 12000 A  -> 0.2 T
  /// dipole current 6000 A
  /// The polarities must match the convention (LHC or DCS2008)
  /// unless the special uniform map was used for MC
  static MagneticField* createFieldMap(Float_t l3Current = -30000., Float_t diCurrent = -6000., Int_t convention = 0,
                                       Bool_t uniform = kFALSE, Float_t beamenergy = 7000, const Char_t* btype = "pp",
                                       const Char_t* path = O2PROTO1_MAGF_CREATEFIELDMAP_DIR);

protected:
  // not supposed to be changed during the run, set only at the initialization via constructor
  void initializeMachineField(BeamType_t btype, Double_t benergy);

  void setBeamType(BeamType_t type)
  {
    mBeamType = type;
  }

  void setBeamEnergy(Float_t energy)
  {
    mBeamEnergy = energy;
  }

protected:
  MagneticWrapperChebyshev* mMeasuredMap; //! Measured part of the field map
  BMap_t mMapType;                        ///< field map type
  Double_t mSolenoid;                     ///< Solenoid field setting
  BeamType_t mBeamType;                   ///< Beam type: A-A (mBeamType=0) or p-p (mBeamType=1)
  Double_t mBeamEnergy;                   ///< Beam energy in GeV

  Int_t mDefaultIntegration;             ///< Default integration method as indicated in Geant
  Int_t mPrecisionInteg;                 ///< Alternative integration method, e.g. for higher precision
  Double_t mMultipicativeFactorSolenoid; ///< Multiplicative factor for solenoid
  Double_t mMultipicativeFactorDipole;   ///< Multiplicative factor for dipole
  Double_t mMaxField;                    ///< Max Field as indicated in Geant
  Bool_t mDipoleOnOffFlag;               ///< Dipole ON/OFF flag

  Double_t mQuadrupoleGradient; ///< Gradient field for inner triplet quadrupoles
  Double_t mDipoleField;        ///< Field value for D1 and D2 dipoles
  Double_t mCompensatorField2C; ///< Side C 2nd compensator field
  Double_t mCompensatorField1A; ///< Side A 1st compensator field
  Double_t mCompensatorField2A; ///< Side A 2nd compensator field

  TNamed mParameterNames; ///< file and parameterization loadad

  static const Double_t sSolenoidToDipoleZ;  ///< conventional Z of transition from L3 to Dipole field
  static const UShort_t sPolarityConvention; ///< convention for the mapping of the curr.sign on main component sign

  FairLogger* mLogger;

    ClassDef(AliceO2::Field::MagneticField, 2) // Class for all Alice MagField wrapper for measured data + Tosca parameterization
};
}
}

#endif
