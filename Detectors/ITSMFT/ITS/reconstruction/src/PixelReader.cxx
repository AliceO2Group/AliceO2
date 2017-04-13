/// \file PixelReader.cxx
/// \brief Implementation of the ITS pixel reader class

#include "ITSReconstruction/PixelReader.h"

using namespace o2::ITS;

//_____________________________________________________
PixelReader::PixelReader()
{
// default constructor
}

//______________________________________________________________________________
Bool_t DigitPixelReader::getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col)
{
  return kTRUE;
}

//______________________________________________________________________________
Bool_t RawPixelReader::getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col)
{
  return kTRUE;
}
