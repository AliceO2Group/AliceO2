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

using namespace AliceO2::Field;


ClassImp(MagFieldFact)


static MagFieldFact gMagFieldFact;

MagFieldFact::MagFieldFact()
  :FairFieldFactory(),
   mFieldPar(NULL)
{
	fCreator=this;
}

MagFieldFact::~MagFieldFact()
{
}

void MagFieldFact::SetParm()
{
  FairRunAna *Run = FairRunAna::Instance();
  FairRuntimeDb *RunDB = Run->GetRuntimeDb();
  mFieldPar = (MagFieldParam*) RunDB->getContainer("MagFieldParam");
}

FairField* MagFieldFact::createFairField()
{ 
  FairField *fMagneticField=0;
  
  if ( !mFieldPar ) {
    FairLogger::GetLogger()->Error(MESSAGE_ORIGIN, "No field parameters available");
    return 0;
  }
  // since we have just 1 field class, we don't need to consider fFieldPar->GetType()
  fMagneticField = new MagneticField(*mFieldPar);
  return fMagneticField;
}


