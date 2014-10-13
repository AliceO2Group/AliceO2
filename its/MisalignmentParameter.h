#ifndef ALICEO2_ITS_MISALIGNMENTPARAMETER_H_
#define ALICEO2_ITS_MISALIGNMENTPARAMETER_H_

#include "FairParGenericSet.h"          // for FairParGenericSet

#include "Rtypes.h"                     // for ClassDef

#include "TArrayD.h"                    // for TArrayD

class FairParamList;

namespace AliceO2 {
namespace ITS {

class MisalignmentParameter : public FairParGenericSet
{
  public:

    MisalignmentParameter(const char* name="MisallignmentParameter",
                                const char* title="Misalignment parameter for O2itsHitProducerIdealMisallign Parameters",
                                const char* context="TestDefaultContext");
    ~MisalignmentParameter(void);
    void clear(void);
    void putParams(FairParamList*);
    Bool_t getParams(FairParamList*);

    TArrayD GetShiftX() {return fShiftX;}
    TArrayD GetShiftY() {return fShiftY;}
    TArrayD GetShiftZ() {return fShiftZ;}
    TArrayD GetRotX() {return fRotX;}
    TArrayD GetRotY() {return fRotY;}
    TArrayD GetRotZ() {return fRotZ;}
    Int_t GetNrOfDetectors() {return fNrOfDetectors;}

  private:

    TArrayD fShiftX; // Array to hold the misalignment in x-direction
    TArrayD fShiftY; // Array to hold the misalignment in y-direction
    TArrayD fShiftZ; // Array to hold the misalignment in z-direction
    TArrayD fRotX; // Array to hold the rotation in x-direction
    TArrayD fRotY; // Array to hold the rotation in y-direction
    TArrayD fRotZ; // Array to hold the rotation in z-direction
    Int_t fNrOfDetectors; // Total number of detectors

    MisalignmentParameter(const MisalignmentParameter&);
    MisalignmentParameter& operator=(const MisalignmentParameter&);

    ClassDef(MisalignmentParameter,1)
};
}
}

#endif
