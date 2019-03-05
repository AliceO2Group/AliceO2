#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TTree.h>
#include <TFile.h>
#include <TStopwatch.h>
#include <FairLogger.h>
#include <vector>
#include <string>

#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#endif

// example of ITS raw data decoding
// Data can be prepared from the MC digits using run_digi2raw_its.C

void run_rawdecoding_its(std::string inpName = "rawits.bin", int verbose = 0)
{

  o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS> rawReader;
  rawReader.openInput(inpName); // RS

  o2::ITSMFT::ChipPixelData chipData;
  rawReader.setVerbosity(verbose);
  TStopwatch sw;
  sw.Start();
  int64_t countDig = 0, countChip = 0;
  while (rawReader.getNextChipData(chipData)) {
    countDig += chipData.getData().size();
    countChip++;
    if (verbose >= 10) {
      chipData.print();
    }
  }
  sw.Stop();

  const auto& MAP = rawReader.getMapping();
  for (int ir = 0; ir < MAP.getNRUs(); ir++) {
    const auto& ruStat = rawReader.getRUDecodingStatSW(ir);
    if (ruStat.nPackets) {
      printf("\nStatistics for RU%3d (HWID:0x%4x)\n", ir, MAP.RUSW2HW(ir));
      ruStat.print();
    }
  }
  rawReader.getDecodingStat().print();
  printf("\n\nDecoded %ld non-empty chips with %ld digits\n", countChip, countDig);

  sw.Print();
}
