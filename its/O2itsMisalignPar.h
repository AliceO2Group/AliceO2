#ifndef O2ITSMISSALLIGNPAR_H
#define O2ITSMISSALLIGNPAR_H

#include "FairParGenericSet.h"          // for FairParGenericSet

#include "Rtypes.h"                     // for ClassDef

#include "TArrayD.h"                    // for TArrayD

class FairParamList;

class O2itsMisalignPar : public FairParGenericSet
{
  public:

    O2itsMisalignPar(const char* name="O2itsMissallignPar",
                                const char* title="Missalignment parameter for O2itsHitProducerIdealMissallign Parameters",
                                const char* context="TestDefaultContext");
    ~O2itsMisalignPar(void);
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

    O2itsMisalignPar(const O2itsMisalignPar&);
    O2itsMisalignPar& operator=(const O2itsMisalignPar&);

    ClassDef(O2itsMisalignPar,1)
};

#endif
