#include "AliVTrdTracklet.h"

const Float_t AliVTrdTracklet::fgkBinWidthY   = 160e-4; // 160 um
const Float_t AliVTrdTracklet::fgkBinWidthDy  = 140e-4; // 140 um
const Float_t AliVTrdTracklet::fgkDriftLength = 3.;     //   3 cm

AliVTrdTracklet::AliVTrdTracklet() :
  TObject()
{
  // default constructor

}

AliVTrdTracklet::AliVTrdTracklet(const AliVTrdTracklet& rhs) :
  TObject(rhs)
{
  // copy constructor

}

AliVTrdTracklet& AliVTrdTracklet::operator=(const AliVTrdTracklet& rhs)
{
  // assignment operator

  if (&rhs != this)
    TObject::operator=(rhs);

  return *this;
}

void AliVTrdTracklet::Copy(TObject &rhs) const
{
  // copy

  TObject::Copy(rhs);
}
