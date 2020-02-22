#include <Rtypes.h>
#include <iostream>

#include "FairLogger.h"
#include "FT0Simulation/Digits2Raw.h"
#include "DataFormatsFT0/RawEventData.h"
#include "DataFormatsFT0/LookUpTable.h"

void FT0digits2raw()
{
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogScreenLevel("DEBUG");
  o2::ft0::Digits2Raw mreader("ft0raw.bin", "ft0digits.root");
}
