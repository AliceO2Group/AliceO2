#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>
#include <string>
#include "TTree.h"
#include "TFile.h"
#include "TOFCalibration/CalibTOFapi.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#endif

void testCCDB()
{

  // macro to populate CCDB for TOF from a "dummy" LHCphase object, and the timeSlewing (+ offset + fineSlewing + problematic) from Run2 latest OCDB objects

  o2::dataformats::CalibLHCphaseTOF* mLHCphaseObj = new o2::dataformats::CalibLHCphaseTOF(); ///< LHCPhase to be written in the output
  mLHCphaseObj->addLHCphase(1567771312, 12);                                                 // "random" numbers, in ps
  mLHCphaseObj->addLHCphase(1567771312 + 86400, 65);                                         // "random" numbers, in ps

  o2::dataformats::CalibTimeSlewingParamTOF* mTimeSlewingObj = new o2::dataformats::CalibTimeSlewingParamTOF(); ///< Time Slewing object to be written in the output
  TFile* f = new TFile("TranslateFromRun2/outputCCDBfromOCDB.root");
  TTree* t = (TTree*)f->Get("tree");
  t->SetBranchAddress("CalibTimeSlewingParamTOF", &mTimeSlewingObj);
  t->GetEvent(0);

  std::map<std::string, std::string> metadataLHCphase;                                        // can be empty
  o2::tof::CalibTOFapi api("http://ccdb-test.cern.ch:8080");                                  // or http://localhost:8080 for a local installation
  api.writeLHCphase(mLHCphaseObj, metadataLHCphase, 1567771312000, 1567771312000 + 86400000); // 06/09/2019 @ 12:01:52am (UTC), till 10/09/2019 @ 12:01:52am (UTC)

  std::map<std::string, std::string> metadataChannelCalib;                         // can be empty
  api.writeTimeSlewingParam(mTimeSlewingObj, metadataChannelCalib, 1567771312000); // 06/09/2019 @ 12:01am (UTC), contains both offset and time slewing

  return;
}
