/// \file Point.h
/// \brief Definition of the Point class
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#ifndef ALICEO2_MFT_POINT_H_
#define ALICEO2_MFT_POINT_H_

#include "FairMCPoint.h"

namespace AliceO2 {
namespace MFT {

class Point : public FairMCPoint
{

 public:

  Point();
  Point(Int_t trackID, Int_t detID, TVector3 pos, TVector3 mom, Double_t time, Double_t length, Double_t eLoss);
  virtual ~Point();

 private:

  ClassDef(Point, 1)

};

}
}

#endif
