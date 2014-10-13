#include "O2itsMisalignPar.h"

#include "FairParamList.h"

//#include <iostream>

ClassImp(O2itsMisalignPar)

O2itsMisalignPar ::O2itsMisalignPar(const char* name,
    const char* title,
    const char* context)
  : FairParGenericSet(name,title,context),
    fShiftX(),
    fShiftY(),
    fShiftZ(),
    fRotX(),
    fRotY(),
    fRotZ(),
    fNrOfDetectors(0)
{
}

O2itsMisalignPar::~O2itsMisalignPar(void)
{
}

void O2itsMisalignPar::clear(void)
{
}

void O2itsMisalignPar::putParams(FairParamList* l)
{
  if (!l) { return; }

  l->add("NrOfDetectors", fNrOfDetectors);
  l->add("ShiftX", fShiftX);
  l->add("ShiftY", fShiftY);
  l->add("ShiftZ", fShiftZ);
  l->add("RotationX", fRotX);
  l->add("RotationY", fRotY);
  l->add("RotationZ", fRotZ);

}

Bool_t O2itsMisalignPar::getParams(FairParamList* l)
{
  if (!l) { return kFALSE; }

  if ( ! l->fill("NrOfDetectors", &fNrOfDetectors) ) { return kFALSE; }

  fShiftX.Set(fNrOfDetectors);
  if ( ! l->fill("ShiftX", &fShiftX )) { return kFALSE; }

  fShiftY.Set(fNrOfDetectors);
  if ( ! l->fill("ShiftY", &fShiftY )) { return kFALSE; }

  fShiftZ.Set(fNrOfDetectors);
  if ( ! l->fill("ShiftZ", &fShiftZ )) { return kFALSE; }

  fRotX.Set(fNrOfDetectors);
  if ( ! l->fill("RotationX", &fRotX )) { return kFALSE; }

  fRotY.Set(fNrOfDetectors);
  if ( ! l->fill("RotationY", &fRotY )) { return kFALSE; }

  fRotZ.Set(fNrOfDetectors);
  if ( ! l->fill("RotationZ", &fRotZ )) { return kFALSE; }

  return kTRUE;
}
