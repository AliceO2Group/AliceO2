/// \file MagFieldFact.h
/// \brief Definition of the MagFieldFact: factory for ALIDE mag. field from MagFieldParam
/// \author ruben.shahoyan@cern.ch


#ifndef ALICEO2_FIELD_MAGFIELDFACT_H
#define ALICEO2_FIELD_MAGFIELDFACT_H

#include "FairFieldFactory.h"
class FairField;

namespace o2 {
  namespace field {
    
class MagFieldParam;

class MagFieldFact : public FairFieldFactory 
{

 public:
  MagFieldFact();
  ~MagFieldFact() override;
  FairField* createFairField() override;
  void SetParm() override;
  
 protected:
  MagFieldParam* mFieldPar;
  
 private:
  MagFieldFact(const MagFieldFact&);
  MagFieldFact& operator=(const MagFieldFact&);

  ClassDefOverride(MagFieldFact,1)
};

}
}

#endif
