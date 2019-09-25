/** 
    It is mandatory that the function returns a FairGenerator* 
    whereas there are no restrictions on the function name
    and the arguments to the function prototype.

    FairGenerator *extgen(Int_t aPDG = 211.);
    
**/

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "FairGenerator.h"
#include "FairPrimaryGenerator.h"
#include <iostream>
#endif

class MyGenerator : public FairGenerator
{
 public:
  MyGenerator(Int_t aPDG) : FairGenerator("MyGenerator"), mPDG(aPDG){};
  Bool_t ReadEvent(FairPrimaryGenerator* primGen) override
  {
    primGen->AddTrack(mPDG, 0.5, 0.5, 0., 0., 0., 0.);
    return kTRUE;
  };

 private:
  Int_t mPDG;
};

FairGenerator*
  extgen(Int_t aPDG = 211)
{
  std::cout << "This is a template function for an custom generator" << std::endl;
  auto gen = new MyGenerator(aPDG);
  return gen;
}
