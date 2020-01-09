#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TStopwatch.h>
#include "DetectorsBase/GeometryManager.h"
#include "MFTReconstruction/ClustererTask.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "FairLogger.h"
#endif

// Clusterization avoiding FairRunAna management.
// Works both with MC digits and with "raw" data (in this case the last argument must be
// set to true). The raw data should be prepared beforeahand from the MC digits using e.g.
// o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingMFT> reader;
// reader.convertDigits2Raw("dig.raw","o2dig.root","o2sim","MFTDigit");
//
// Use for MC mode:
// root -b -q run_clus_itsSA.C+\(\"o2clus_its.root\",\"o2dig.root\"\) 2>&1 | tee clusSA.log
//
// Use for RAW mode:
// root -b -q run_clus_itsSA.C+\(\"o2clus_its.root\",\"dig.raw\"\) 2>&1 | tee clusSARAW.log

void run_clus_mftSA(std::string outputfile, // output file name
                    std::string inputfile,  // input file name (root or raw)
                    bool raw = false,       // flag if this is raw data
                    float strobe = 6000.)   // strobe length of ALPIDE readout
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  TStopwatch timer;
  o2::base::GeometryManager::loadGeometry(); // needed provisionary, only to write full clusters

  // Setup clusterizer
  Bool_t useMCTruth = kTRUE;  // kFALSE if no comparison with MC needed
  o2::mft::ClustererTask* clus = new o2::mft::ClustererTask(useMCTruth, raw);

  // Mask fired pixels separated by <= this number of BCs (for overflow pixels).
  // In continuos mode strobe lenght should be used, in triggered one: signal shaping time (~7mus)
  clus->getClusterer().setMaxBCSeparationToMask(strobe / o2::constants::lhc::LHCBunchSpacingNS + 10);
  clus->getClusterer().setWantFullClusters(true);    // require clusters with coordinates and full pattern
  clus->getClusterer().setWantCompactClusters(true); // require compact clusters with patternID

  clus->run(inputfile, outputfile);

  timer.Stop();
  timer.Print();
}
