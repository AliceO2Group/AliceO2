#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>
#include <TStopwatch.h>
#include "ITSMFTReconstruction/RawPixelReader.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#endif

void run_digi2raw_its(std::string outFile = "its.raw", std::string inpFile = "o2dig.root",
                      UShort_t modFirst = 0, UShort_t modLast = 0xffff)
{
  TStopwatch sw;
  sw.Start();
  o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS> reader;
  reader.convertDigits2Raw(outFile, inpFile, "o2sim", "ITSDigit", modFirst, modLast);
  sw.Stop();
}
