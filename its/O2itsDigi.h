/*
 * O2itsDigi.h
 *
 *  Created on: 20.07.2012
 *      Author: stockman
 */

#ifndef O2ITSDIGI_H_
#define O2ITSDIGI_H_

#include "FairTimeStamp.h"              // for FairTimeStamp

#include "Riosfwd.h"                    // for ostream
#include "Rtypes.h"                     // for Int_t, etc

#include <iostream>                     // for operator<<, basic_ostream, etc

class O2itsDigi : public FairTimeStamp
{
  public:
    O2itsDigi();
    O2itsDigi(Int_t x, Int_t y, Int_t z, Double_t timeStamp);
    virtual ~O2itsDigi();

    void SetXYZ(Int_t x, Int_t y, Int_t z) { SetX(x); SetY(y); SetZ(z);}
    void SetX(Int_t x)  { fX=x; }
    void SetY(Int_t y)  { fY=y; }
    void SetZ(Int_t z)  { fZ=z; }

    Int_t GetX() const  {return fX;}
    Int_t GetY() const  {return fY;}
    Int_t GetZ() const  {return fZ;}

    virtual bool equal(FairTimeStamp* data) {
      O2itsDigi* myDigi = dynamic_cast <O2itsDigi*> (data);
      if (myDigi != 0) {
        if (fX == myDigi->GetX())
          if (fY == myDigi->GetY())
            if (fZ == myDigi->GetZ()) {
              return true;
            }
      }
      return false;
    }

    virtual bool operator<(const O2itsDigi& myDigi) const {
      if (fX < myDigi.GetX()) { return true; }
      else if (fX > myDigi.GetX()) { return false; }
      if (fY < myDigi.GetY()) { return true; }
      else if (fY > myDigi.GetY()) { return false; }
      if (fZ < myDigi.GetZ()) { return true; }
      else if (fZ > myDigi.GetZ()) { return false; }
      return false;
    }

    friend std::ostream& operator<<(std::ostream& out, O2itsDigi& digi) {
      out << "O2itsDigi at:"
          << " (" << digi.GetX() << "/" << digi.GetY() << "/" << digi.GetZ() << ") "
          << " with TimeStamp: " << digi.GetTimeStamp()
          << std::endl;
      return out;
    }

  private:
    Int_t fX;
    Int_t fY;
    Int_t fZ;


    ClassDef(O2itsDigi,1);
};

#endif
