#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TStopwatch.h>
#include "DetectorsBase/GeometryManager.h"
#include "ITSReconstruction/ClustererTask.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "FairLogger.h"
#endif

// Clusterization avoiding FairRunAna management.
// Works both with MC digits and with "raw" data (in this case the last argument must be
// set to true). The raw data should be prepared beforeahand from the MC digits using e.g.
// o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingITS> reader;
// reader.convertDigits2Raw("dig.raw","o2dig.root","o2sim","ITSDigit");
//
// Use for MC mode:
// root -b -q run_clus_itsSA.C+\(\"o2clus_its.root\",\"o2dig.root\"\) 2>&1 | tee clusSA.log
//
// Use for RAW mode:
// root -b -q run_clus_itsSA.C+\(\"o2clus_its.root\",\"dig.raw\"\) 2>&1 | tee clusSARAW.log

void run_clus_itsSA(std::string outputfile, std::string inputfile, bool raw = false)
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  TStopwatch timer;
  o2::Base::GeometryManager::loadGeometry(); // needed provisionary, only to write full clusters

  // Setup clusterizer
  Bool_t useMCTruth = kTRUE;  // kFALSE if no comparison with MC needed
  Bool_t entryPerROF = kTRUE; // write single tree entry for every ROF. If false, just 1 entry will be saved
  o2::ITS::ClustererTask* clus = new o2::ITS::ClustererTask(useMCTruth, raw);
  clus->getClusterer().setMaskOverflowPixels(true);  // set this to false to switch off masking
  clus->getClusterer().setWantFullClusters(true);    // require clusters with coordinates and full pattern
  clus->getClusterer().setWantCompactClusters(true); // require compact clusters with patternID

  clus->run(inputfile, outputfile, entryPerROF);

  timer.Stop();
  timer.Print();
}
