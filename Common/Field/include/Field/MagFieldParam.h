/// \file MagFieldParam.h
/// \brief Definition of the MagFieldParam: container for ALICE mag. field parameters
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGFIELDPARAM_H_
#define ALICEO2_FIELD_MAGFIELDPARAM_H_

#include "FairParGenericSet.h"
#include <TString.h>

class FairParamList;


namespace AliceO2 {
  namespace Field {

class MagneticField;
    
class MagFieldParam : public FairParGenericSet
{
  public:
    enum BMap_t
    {
      k2kG, k5kG, k5kGUniform, kNFieldTypes
    };
    enum BeamType_t
    {
        kNoBeamField, kBeamTypepp, kBeamTypeAA, kBeamTypepA, kBeamTypeAp
    };

    MagFieldParam(const char* name="", const char* title="", const char* context="");

    void SetParam(const MagneticField* field);
    
    BMap_t     GetMapType()                   const {return mMapType;}
    BeamType_t GetBeamType()                  const {return mBeamType;}
    Int_t                     GetDefInt()     const {return mDefaultIntegration;}
    Double_t                  GetFactorSol()  const {return mFactorSol;}
    Double_t                  GetFactorDip()  const {return mFactorDip;}
    Double_t                  GetBeamEnergy() const {return mBeamEnergy;}
    Double_t                  GetMaxField()   const {return mMaxField;}
    const char*               GetMapPath()    const {return mMapPath.Data();}

    virtual void   putParams(FairParamList* list);
    virtual Bool_t getParams(FairParamList* list);
    
  protected:
    BMap_t     mMapType;  ///< map type ID
    BeamType_t mBeamType; ///< beam type ID
    Int_t    mDefaultIntegration;        ///< field integration type for MC
    Double_t mFactorSol;                 ///< solenoid current factor
    Double_t mFactorDip;                 ///< dipole current factor
    Double_t mBeamEnergy;                ///< beam energy
    Double_t mMaxField;                  ///< max field for geant
    TString  mMapPath;                   ///< path to map file
    
    ClassDef(MagFieldParam,1)
};
}
}

#endif
