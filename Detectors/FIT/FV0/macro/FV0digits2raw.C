#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <Rtypes.h>
#include <iostream>

#include "FairLogger.h"
#include "FV0Simulation/Digits2Raw.h"
#include "DataFormatsFV0/RawEventData.h"
#include "DataFormatsFV0/LookUpTable.h"

void FV0digits2raw()
{
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogScreenLevel("DEBUG");
  logger->SetLogScreenLevel("INFO");
  o2::fv0::Digits2Raw mreader("fv0raw.bin", "fv0digits.root");
  RawEventData::print();
}

#endif