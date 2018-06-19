#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include <TStopwatch.h>
//#include "DataFormatsParameters/GRPObject.h"
#include "FairFileSource.h"
#include "FairLogger.h"
#include "FairRunAna.h"
//#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"

#include "PHOSReconstruction/ClustererTask.h"
#endif

void run_clu_phos(std::string outputfile = "o2clu.root", std::string inputfile = "o2dig.root",
                  std::string paramfile = "AliceO2_TGeant3.phos.params_10.root")
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  //  logger->SetLogScreenLevel("INFO");
  logger->SetLogScreenLevel("DEBUG");

  // Setup timer
  TStopwatch timer;

  // gDebug=1;

  // Setup FairRoot analysis manager
  FairRunAna* fRun = new FairRunAna();
  FairFileSource* fFileSource = new FairFileSource(inputfile);
  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetSource(fFileSource);
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(outputfile.data());

  // Setup Runtime DB
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.data());
  rtdb->setFirstInput(parInput1);

  // Setup digitizer
  o2::phos::ClustererTask* clu = new o2::phos::ClustererTask();
  fRun->AddTask(clu);

  fRun->Init();
  timer.Start();
  fRun->Run();

  // Extract the maximal used memory an add is as Dart measurement
  // This line is filtered by CTest and the value send to CDash
  FairSystemInfo sysInfo;
  Float_t maxMemory = sysInfo.GetMaxMemory();
  std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage = ctime / rtime;
  cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
  cout << cpuUsage;
  cout << "</DartMeasurement>" << std::endl;
  cout << std::endl;
  std::cout << "Macro finished succesfully" << std::endl;

  std::cout << std::endl;
  std::cout << "Output file is " << outputfile << std::endl;
  // std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << std::endl;
}
