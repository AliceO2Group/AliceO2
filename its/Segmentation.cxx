/// \file Segmentation.cxx
/// \brief Implementation of the Segmentation class

#include <TF1.h>
#include "Segmentation.h"

using namespace AliceO2::ITS;

ClassImp(Segmentation)

Segmentation::Segmentation()
    : mDx(0)
    , mDz(0)
    , mDy(0)
    , mCorrection(0)
{
}

Segmentation::~Segmentation()
{
  if (mCorrection) {
    delete mCorrection;
  }
}

void Segmentation::Copy(TObject& obj) const
{
  // copy this to obj
  ((Segmentation&)obj).mDz = mDz;
  ((Segmentation&)obj).mDx = mDx;
  ((Segmentation&)obj).mDy = mDy;

  if (mCorrection) {
    ((Segmentation&)obj).mCorrection = new TF1(*mCorrection); // make a proper copy
  }
  else {
    ((Segmentation&)obj).mCorrection = 0;
  }
}

Segmentation& Segmentation::operator=(const Segmentation& source)
{
  // Operator =
  if (this != &source) {
    source.Copy(*this);
  }
  return *this;
}

Segmentation::Segmentation(const Segmentation& source)
    : TObject(source)
    , mDx(0)
    , mDz(0)
    , mDy(0)
    , mCorrection(0)
{
  // copy constructor
  source.Copy(*this);
}
