// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagneticField.h
/// \brief Definition of the MagF class
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGNETICFIELD_H_
#define ALICEO2_FIELD_MAGNETICFIELD_H_

#include "FairField.h"         // for FairField
#include "Field/MagFieldParam.h"
#include "Field/MagneticWrapperChebyshev.h" // for MagneticWrapperChebyshev
#include "Field/MagFieldFast.h"
#include "TSystem.h"
#include "Rtypes.h"            // for Double_t, Char_t, Int_t, Float_t, etc
#include "TNamed.h"            // for TNamed
#include <memory>              // for str::unique_ptr

class FairLogger;  // lines 14-14
class FairParamList;

namespace o2 { namespace field { class MagneticWrapperChebyshev; }}  // lines 19-19
namespace o2 {
namespace field {


/// Interface between the TVirtualMagField and MagneticWrapperChebyshev: wrapper to the set of magnetic field data +
/// Tosca
/// parametrization by Chebyshev polynomials
class MagneticField : public FairField
{

  public:
    enum PolarityConvention_t
    {
        kConvLHC, kConvDCS2008, kConvMap2005
    };
    enum
    {
        kOverrideGRP = BIT(14)
    }; // don't recreate from GRP if set

    /// Default constructor
    MagneticField();

    /// Initialize the field with Geant integration option "integ" and max field "fmax",
    /// Impose scaling of parameterized L3 field by factorSol and of dipole by factorDip.
    /// The "be" is the energy of the beam in GeV/nucleon
    MagneticField(const char *name, const char *title, Double_t factorSol = 1., Double_t factorDip = 1.,
                  MagFieldParam::BMap_t maptype = MagFieldParam::k5kG,
		  MagFieldParam::BeamType_t btype = MagFieldParam::kBeamTypepp,
		  Double_t benergy = -1, Int_t integ = 2,
                  Double_t fmax = 15, const std::string path = std::string(gSystem->ExpandPathName("$(O2_ROOT)"))
		  + std::string("/share/Common/maps/mfchebKGI_sym.root")
    );


    MagneticField(const MagFieldParam& param);

    MagneticField &operator=(const MagneticField &src);

    /// Default destructor
    ~MagneticField() override = default;

    /// real field creation is here
    void CreateField();

    /// allow fast field param
    void        AllowFastField(bool v=true);
    
    /// Virtual methods from FairField

    /// X component, avoid using since slow
    Double_t GetBx(Double_t x, Double_t y, Double_t z) override {
      double xyz[3]={x,y,z},b[3];
      MagneticField::Field(xyz,b);
      return b[0];
    } 

    /// Y component, avoid using since slow
    Double_t GetBy(Double_t x, Double_t y, Double_t z) override {
      double xyz[3]={x,y,z},b[3];
      MagneticField::Field(xyz,b);
      return b[1];
    }

    /// Z component
    Double_t GetBz(Double_t x, Double_t y, Double_t z) override {
      double xyz[3]={x,y,z};
      return getBz(xyz); 
    } 

    /// Method to calculate the field at point xyz
    /// Main interface from TVirtualMagField used in simulation
    void Field(const Double_t* __restrict__ point, Double_t* __restrict__ bField) override;

    /// 3d field query alias for Alias Method to calculate the field at point xyz
    void GetBxyz(const Double_t p[3], Double_t* b) override { MagneticField::Field(p,b); }

    /// Fill Paramater
    void FillParContainer() override;
    
    /// Method to calculate the integral_0^z of br,bt,bz
    void getTPCIntegral(const Double_t *xyz, Double_t *b) const;

    /// Method to calculate the integral_0^z of br,bt,bz
    void getTPCRatIntegral(const Double_t *xyz, Double_t *b) const;

    /// Method to calculate the integral_0^z of br,bt,bz in cylindrical coordinates ( -pi<phi<pi convention )
    void getTPCIntegralCylindrical(const Double_t *rphiz, Double_t *b) const;

    /// Method to calculate the integral_0^z of bx/bz,by/bz and (bx/bz)^2+(by/bz)^2 in
    /// cylindrical coordinates ( -pi<phi<pi convention )
    void getTPCRatIntegralCylindrical(const Double_t *rphiz, Double_t *b) const;

    /// Method to calculate the field at point xyz
    Double_t getBz(const Double_t *xyz) const;

    MagneticWrapperChebyshev *getMeasuredMap() const { return mMeasuredMap.get();}
    
    /// get fast field direct pointer
    const MagFieldFast* getFastField() const {return mFastField.get();}
    
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
      return getFactorSolenoid() * (mMapType == MagFieldParam::k2kG ? 12000 : 30000);
    }

    Double_t getCurrentDipole() const
    {
      return getFactorDipole() * 6000;
    }

    Bool_t IsUniform() const
    {
      return mMapType == MagFieldParam::k5kGUniform;
    }

    void MachineField(const Double_t * __restrict__ x, Double_t * __restrict__ b) const;

    MagFieldParam::BMap_t getMapType() const
    {
      return mMapType;
    }

    MagFieldParam::BeamType_t getBeamType() const
    {
      return mBeamType;
    }

    /// Returns beam type in text form
    const char *getBeamTypeText() const;

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
      return mSolenoid;
    }

    Char_t *getDataFileName() const
    {
      return (Char_t *) mParameterNames.GetName();
    }

    Char_t *getParameterName() const
    {
      return (Char_t *) mParameterNames.GetTitle();
    }

    void setDataFileName(const Char_t *nm)
    {
      mParameterNames.SetName(nm);
    }

    void setParameterName(const Char_t *nm)
    {
      mParameterNames.SetTitle(nm);
    }

    /// Prints short or long info
    void Print(Option_t *opt) const override;

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
    static MagneticField *createFieldMap(Float_t l3Current = -30000., Float_t diCurrent = -6000., Int_t convention = 0,
                                         Bool_t uniform = kFALSE, Float_t beamenergy = 7000, const Char_t *btype = "pp",
                                         const std::string path = std::string(gSystem->Getenv("VMCWORKDIR")) +
                                                                  std::string("/Common/maps/mfchebKGI_sym.root")
    );

  protected:
    // not supposed to be changed during the run, set only at the initialization via constructor
    void initializeMachineField(MagFieldParam::BeamType_t btype, Double_t benergy);

    void setBeamType(MagFieldParam::BeamType_t type)
    {
      mBeamType = type;
    }

    void setBeamEnergy(Float_t energy)
    {
      mBeamEnergy = energy;
    }

  private:
    std::unique_ptr<MagneticWrapperChebyshev> mMeasuredMap; //! Measured part of the field map
    std::unique_ptr<MagFieldFast>             mFastField; // ! optional fast parametrization
    MagFieldParam::BMap_t mMapType;         ///< field map type
    Double_t mSolenoid;                     ///< Solenoid field setting
    MagFieldParam::BeamType_t mBeamType;    ///< Beam type: A-A (mBeamType=0) or p-p (mBeamType=1)
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

    TNamed mParameterNames; ///< file and parameterization loaded

    static const Double_t sSolenoidToDipoleZ;  ///< conventional Z of transition from L3 to Dipole field
    static const UShort_t sPolarityConvention; ///< convention for the mapping of the curr.sign on main component sign

    FairLogger *mLogger;

    MagneticField(const MagneticField &src);


    
    ClassDefOverride(o2::field::MagneticField,
    3) // Class for all Alice MagField wrapper for measured data + Tosca parameterization
};
}
}

#endif
