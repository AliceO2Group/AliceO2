#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <Rtypes.h>
#include <iostream>

#include "FairLogger.h"
#include "FT0Simulation/Digits2Raw.h"
#include "DataFormatsFT0/RawEventData.h"
#include "DataFormatsFT0/LookUpTable.h"
#endif

void FV0raw2digits()
{
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogScreenLevel("DEBUG");
  o2::fv0::ReadRaw mreader("fv0.raw", "fv0digitsFromRaw.root");
}
