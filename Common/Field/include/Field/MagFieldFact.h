/// \file MagFieldFact.h
/// \brief Definition of the MagFieldFact: factory for ALIDE mag. field from MagFieldParam
/// \author ruben.shahoyan@cern.ch


#ifndef ALICEO2_FIELD_MAGFIELDFACT_H
#define ALICEO2_FIELD_MAGFIELDFACT_H

#include "FairFieldFactory.h"
class FairField;

namespace AliceO2 {
  namespace Field {
    
class MagFieldParam;

class MagFieldFact : public FairFieldFactory 
{

 public:
  MagFieldFact();
  virtual ~MagFieldFact();
  virtual FairField* createFairField();
  virtual void SetParm();
  
 protected:
  MagFieldParam* fFieldPar;
  
 private:
  MagFieldFact(const MagFieldFact&);
  MagFieldFact& operator=(const MagFieldFact&);

  ClassDef(MagFieldFact,1)
};

}
}

#endif
