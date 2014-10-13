/*
 * O2itsDigi.cxx
 *
 *  Created on: 20.07.2012
 *      Author: stockman
 */

#include "O2itsDigi.h"

ClassImp(O2itsDigi);

O2itsDigi::O2itsDigi():
  FairTimeStamp(), fX(0), fY(0), fZ(0)
{
}


O2itsDigi::O2itsDigi(Int_t x, Int_t y, Int_t z, Double_t timeStamp):
  FairTimeStamp(timeStamp), fX(x), fY(y), fZ(z)
{
}

O2itsDigi::~O2itsDigi()
{
}



