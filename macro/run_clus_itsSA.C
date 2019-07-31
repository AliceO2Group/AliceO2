#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TStopwatch.h>
#include "DetectorsBase/GeometryManager.h"
#include "ITSReconstruction/ClustererTask.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"
#include "FairLogger.h"
#endif

// Clusterization avoiding FairRunAna management.
// Works both with MC digits and with "raw" data (in this case the last argument must be
// set to true). The raw data should be prepared beforeahand from the MC digits using e.g.
// o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS> reader;
// reader.convertDigits2Raw("dig.raw","o2dig.root","o2sim","ITSDigit");
//
// Use for MC mode:
// root -b -q run_clus_itsSA.C+\(\"o2clus_its.root\",\"o2dig.root\"\) 2>&1 | tee clusSA.log
//
// Use for RAW mode:
// root -b -q run_clus_itsSA.C+\(\"o2clus_its.root\",\"dig.raw\"\) 2>&1 | tee clusSARAW.log
//
// Use of topology dictionary: flag withDicitonary -> true
// A dictionary must be generated with the macro CheckTopologies.C

void run_clus_itsSA(std::string inputfile = "rawits.bin", // output file name
                    std::string outputfile = "clr.root",  // input file name (root or raw)
                    bool raw = true,                      // flag if this is raw data
                    float strobe = -1.,                   // strobe length in ns of ALPIDE readout, if <0, get automatically
                    bool withDictionary = false, std::string dictionaryfile = "complete_dictionary.bin")
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  TStopwatch timer;
  o2::base::GeometryManager::loadGeometry(); // needed provisionary, only to write full clusters

  // Setup clusterizer
  Bool_t useMCTruth = kTRUE;  // kFALSE if no comparison with MC needed
  Bool_t entryPerROF = kTRUE; // write single tree entry for every ROF. If false, just 1 entry will be saved
  o2::its::ClustererTask* clus = new o2::its::ClustererTask(useMCTruth, raw);
  if (withDictionary) {
    clus->loadDictionary(dictionaryfile.c_str());
  }
  // Mask fired pixels separated by <= this number of BCs (for overflow pixels).
  // In continuos mode strobe lenght should be used, in triggered one: signal shaping time (~7mus)
  if (strobe < 0) {
    const auto& dgParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    strobe = dgParams.roFrameLength;
  }
  clus->getClusterer().setMaxBCSeparationToMask(strobe / o2::constants::lhc::LHCBunchSpacingNS + 10);
  clus->getClusterer().setWantFullClusters(true);    // require clusters with coordinates and full pattern
  clus->getClusterer().setWantCompactClusters(withDictionary); // require compact clusters with patternID

  clus->getClusterer().print();
  clus->run(inputfile, outputfile, entryPerROF);

  timer.Stop();
  timer.Print();
}
