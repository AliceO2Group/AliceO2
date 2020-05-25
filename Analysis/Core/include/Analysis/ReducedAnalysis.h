#ifndef REDUCEDANALYSIS_H
#define REDUCEDANALYSIS_H

#include <TObject.h>
#include <TNamed.h>
#include <TString.h> 

//________________________________________________________________
class ReducedAnalysis : public TObject {
  
public:
   
  ReducedAnalysis();
  ReducedAnalysis(const char* name, const char* title);
  virtual ~ReducedAnalysis();
  
  virtual void Init();
  virtual void Process();
  virtual void Finish();
  
protected:
  ReducedAnalysis(const ReducedAnalysis& task);             
  ReducedAnalysis& operator=(const ReducedAnalysis& task);      
  
  uint64_t fEventCounter;   // event counter
  
  ClassDef(ReducedAnalysis, 1)
};

#endif
