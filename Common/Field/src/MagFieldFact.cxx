#include "Field/MagFieldFact.h"
#include "Field/MagFieldParam.h"
#include "Field/MagneticField.h"
#include "FairRunAna.h"
#include "FairRuntimeDb.h"
#include "FairField.h"

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

using namespace o2::field;


ClassImp(MagFieldFact)


static MagFieldFact gMagFieldFact;

MagFieldFact::MagFieldFact()
  :FairFieldFactory(),
   mFieldPar(nullptr)
{
	fCreator=this;
}

MagFieldFact::~MagFieldFact()
= default;

void MagFieldFact::SetParm()
{
  FairRunAna *Run = FairRunAna::Instance();
  FairRuntimeDb *RunDB = Run->GetRuntimeDb();
  mFieldPar = (MagFieldParam*) RunDB->getContainer("MagFieldParam");
}

FairField* MagFieldFact::createFairField()
{ 
  FairField *fMagneticField=nullptr;
  
  if ( !mFieldPar ) {
    FairLogger::GetLogger()->Error(MESSAGE_ORIGIN, "No field parameters available");
    return nullptr;
  }
  // since we have just 1 field class, we don't need to consider fFieldPar->GetType()
  fMagneticField = new MagneticField(*mFieldPar);
  return fMagneticField;
}


