#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TStopwatch.h>
#include <fairlogger/Logger.h>
#include <fstream>
#include <string>

#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/RUDecodeData.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTReconstruction/RawPixelReader.h"
#endif

// example of skimming of the CRU data with 128b-padded GBT words and fixed 8KB page
// to 80b GBT words in the pages size corresponding to the real payload

// Initial raw can be prepared from the MC digits using run_digi2raw_its.C

void run_CRUDataSkimming_its(std::string inpName = "rawits.bin",
                             std::string outName = "rawits_skimmed.bin",
                             int nTriggersToCache = 1025, // number of triggers per link to cache (> N 8KB CRU pages per superpage)
                             int verbose = 0)
{

  o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS> rawReader;
  rawReader.openInput(inpName);
  rawReader.setPadding128(false);
  rawReader.setMinTriggersToCache(nTriggersToCache);
  rawReader.setVerbosity(verbose);

  std::fstream outFile(outName, std::ios::out | std::ios::binary);
  o2::itsmft::PayLoadCont outBuffer(1000000); // book 1 MB buffer

  TStopwatch sw, swIO;
  sw.Start();
  swIO.Stop();

  while (rawReader.skimNextRUData(outBuffer)) {
    swIO.Start(false);
    outFile.write((const char*)outBuffer.data(), outBuffer.getSize());
    swIO.Stop();
    outBuffer.clear();
  }

  outFile.close();
  sw.Stop();

  const auto& MAP = rawReader.getMapping();
  for (int ir = 0; ir < MAP.getNRUs(); ir++) {
    for (int il = 0; il < o2::itsmft::RUDecodeData::MaxLinksPerRU; il++) {
      const auto ruStat = rawReader.getRUDecodingStatSW(ir, il);
      if (ruStat && ruStat->nPackets) {
        printf("\nStatistics for RU%3d (HWID:0x%4x) GBTLink%d\n", ir, MAP.RUSW2FEEId(ir, il), il);
        ruStat->print();
      }
    }
  }
  rawReader.getDecodingStat().print();

  printf("Total time spent on skimming: ");
  sw.Print();
  printf("Time spent on writing output: ");
  swIO.Print();
}
