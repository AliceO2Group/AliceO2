/// \file Hit.h
/// \brief Simple hit obtained from points
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#ifndef ALICEO2_MFT_HIT_H_
#define ALICEO2_MFT_HIT_H_

#include "FairHit.h"

class TVector3;

namespace o2 {
namespace MFT {

class Hit : public FairHit
{

 public:
  
  Hit();
  Hit(Int_t detID, TVector3& pos, TVector3& dpos, Int_t mcindex);
  
  ~Hit() override;
  
 private:
  
  Hit(const Hit&);
  Hit operator=(const Hit&);

  ClassDefOverride(Hit,1);

};

}
}

#endif
