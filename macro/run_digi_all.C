#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>

#include "Rtypes.h"
#include "TString.h"
#include "TStopwatch.h"
#include "TGeoManager.h"

#include "FairLogger.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairSystemInfo.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairLinkManager.h"

#include "DataFormatsParameters/GRPObject.h"

#include "TPCSimulation/DigitizerTask.h"
#include "ITSSimulation/DigitizerTask.h"
#endif

int updateITSTPCinGRP(std::string inputGRP, std::string grpName = "GRP");

void run_digi_all(float rate = 100e3, std::string outputfile = "o2dig.root", std::string inputfile = "o2sim.root",
                  std::string paramfile = "o2sim_par.root", std::string inputGRP = "o2sim_grp.root")
{

  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("DEBUG");

  // Setup timer
  TStopwatch timer;

  // Setup FairRoot analysis manager
  FairRunAna* run = new FairRunAna();
  FairFileSource* fFileSource = new FairFileSource(inputfile.data());

  run->SetSource(fFileSource);
  run->SetOutputFile(outputfile.data());

  if (rate > 0) {
    fFileSource->SetEventMeanTime(1.e9 / rate); // is in us
    // update GRP flagging continuously readout detectors
    updateITSTPCinGRP(inputGRP);
  }

  // Needed for TPC
  run->SetUseFairLinks(kTRUE);
  // -- only store the link to the MC track
  //    if commented, also the links to all previous steps will be stored
  FairLinkManager::Instance()->AddIncludeType(0);

  // Setup Runtime DB
  FairRuntimeDb* rtdb = run->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.data());
  rtdb->setFirstInput(parInput1);

  //============================= create digitizers =============================>>>

  o2::ITS::DigitizerTask* digiITS = new o2::ITS::DigitizerTask();
  digiITS->setContinuous(rate > 0);
  digiITS->setFairTimeUnitInNS(1.0);     // tell in which units (wrt nanosecond) FAIR timestamps are
  digiITS->setAlpideROFramLength(5000.); // ALPIDE RO frame in ns
  run->AddTask(digiITS);

  o2::TPC::DigitizerTask* digiTPC = new o2::TPC::DigitizerTask;
  digiTPC->setContinuousReadout(rate > 0);
  digiTPC->setDebugOutput("DigitMCDebug");
  run->AddTask(digiTPC);

  //-----------
  run->Init();
  // ============================================================================<<<

  timer.Start();
  run->Run();

  std::cout << std::endl << std::endl;

  // Extract the maximal used memory an add is as Dart measurement
  // This line is filtered by CTest and the value send to CDash
  FairSystemInfo sysInfo;
  Float_t maxMemory = sysInfo.GetMaxMemory();
  std::cout << R"(<DartMeasurement name="MaxMemory" type="numeric/double">)";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage = ctime / rtime;
  std::cout << R"(<DartMeasurement name="CpuLoad" type="numeric/double">)";
  std::cout << cpuUsage;
  std::cout << "</DartMeasurement>" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "Macro finished succesfully." << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "Output file is " << outputfile.data() << std::endl;
  // std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << std::endl << std::endl;
}

int updateITSTPCinGRP(std::string inputGRP, std::string grpName)
{
  TFile flGRP(inputGRP.data(), "update");
  if (flGRP.IsZombie()) {
    LOG(ERROR) << "Failed to open in update mode " << inputGRP << FairLogger::endl;
    return -10;
  }
  auto grp =
    static_cast<o2::parameters::GRPObject*>(flGRP.GetObjectChecked(grpName.data(), o2::parameters::GRPObject::Class()));
  if (!grp) {
    LOG(ERROR) << "Did not find GRP object named " << inputGRP << FairLogger::endl;
    return -12;
  }

  vector<o2::detectors::DetID> contDet = { o2::detectors::DetID::ITS, o2::detectors::DetID::TPC };

  for (auto det : contDet) {
    if (grp->isDetReadOut(det)) {
      grp->addDetContinuousReadOut(det);
    }
  }

  LOG(INFO) << "Updated GRP in " << inputGRP << " flagging continously read-out detectors" << FairLogger::endl;
  grp->print();
  flGRP.WriteObjectAny(grp, grp->Class(), grpName.data());
  flGRP.Close();
  return 0;
}
