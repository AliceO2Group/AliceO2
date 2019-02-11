#include "AliVTrdTrack.h"

AliVTrdTrack::AliVTrdTrack() :
  TObject()
{
  // default constructor

}

AliVTrdTrack::AliVTrdTrack(const AliVTrdTrack& rhs) :
  TObject(rhs)
{
  // copy constructor

}

AliVTrdTrack& AliVTrdTrack::operator=(const AliVTrdTrack& rhs)
{
  // assignment operator

  if (&rhs != this)
    TObject::operator=(rhs);

  return *this;
}

void AliVTrdTrack::Copy(TObject &rhs) const
{
  // copy

  TObject::Copy(rhs);
}
