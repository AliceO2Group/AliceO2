/// \file Segmentation.cxx
/// \brief Implementation of the Segmentation class

#include "ITSMFTBase/Segmentation.h"
#include "TF1.h" // for TF1

using namespace o2::ITSMFT;

ClassImp(o2::ITSMFT::Segmentation)

  Segmentation::Segmentation()
  : mDx(0), mDz(0), mDy(0), mCorrection(nullptr)
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
  } else {
    ((Segmentation&)obj).mCorrection = nullptr;
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

Segmentation::Segmentation(const Segmentation& source) : TObject(source), mDx(0), mDz(0), mDy(0), mCorrection(nullptr)
{
  // copy constructor
  source.Copy(*this);
}

