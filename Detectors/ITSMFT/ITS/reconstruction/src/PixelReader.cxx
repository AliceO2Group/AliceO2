/// \file PixelReader.cxx
/// \brief Implementation of the ITS pixel reader class

#include <TClonesArray.h>

#include "ITSMFTBase/Digit.h"
#include "ITSReconstruction/PixelReader.h"

using namespace o2::ITS;
using o2::ITSMFT::Digit;

//______________________________________________________________________________
Bool_t DigitPixelReader::getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col, Int_t &label)
{
  if (mIdx >= mDigitArray->GetEntriesFast()) return kFALSE;

  const Digit *d=static_cast<Digit *>(mDigitArray->UncheckedAt(mIdx++));
  id = d->getChipIndex();
  row = d->getRow();
  col = d->getColumn();
  label = d->getLabel(0);
  
  return kTRUE;
}

//______________________________________________________________________________
Bool_t RawPixelReader::getNextFiredPixel(UShort_t &id, UShort_t &row, UShort_t &col, Int_t &label)
{
  return kTRUE;
}
