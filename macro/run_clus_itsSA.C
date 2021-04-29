#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TStopwatch.h>
#include "DetectorsBase/GeometryManager.h"
#include "ITSReconstruction/ClustererTask.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsCommonDataFormats/NameConf.h"
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

void run_clus_itsSA(std::string inputfile = "rawits.bin", // input file name
                    std::string outputfile = "clr.root",  // output file name (root or raw)
                    bool raw = true,                      // flag if this is raw data
                    int strobeBC = -1,                    // strobe length in BC for masking, if <0, get automatically (assume cont. readout)
                    std::string dictionaryfile = "",
                    bool withPatterns = true)
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  TStopwatch timer;

  // Setup clusterizer
  Bool_t useMCTruth = kTRUE;  // kFALSE if no comparison with MC needed
  o2::its::ClustererTask* clus = new o2::its::ClustererTask(useMCTruth, raw);
  clus->setMaxROframe(2 << 21); // about 3 cluster files per a raw data chunk

  if (dictionaryfile.empty()) {
    dictionaryfile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  std::ifstream file(dictionaryfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictionaryfile.c_str();
    clus->loadDictionary(dictionaryfile.c_str());
  } else {
    LOG(INFO) << "Running without dictionary !";
  }

  // Mask fired pixels separated by <= this number of BCs (for overflow pixels).
  // In continuos mode strobe lenght should be used, in triggered one: signal shaping time (~7mus)
  if (strobeBC < 0) {
    const auto& dgParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    strobeBC = dgParams.roFrameLengthInBC;
  }
  clus->getClusterer().setMaxBCSeparationToMask(strobeBC + 10);

  clus->getClusterer().print();
  clus->run(inputfile, outputfile);

  timer.Stop();
  timer.Print();
}
