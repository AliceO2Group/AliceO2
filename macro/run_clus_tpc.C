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

#include "TPCReconstruction/ClustererTask.h"
#endif

void run_clus_tpc(std::string outputfile = "o2clus_tpc.root", std::string inputfile = "o2dig.root",
                  std::string paramfile = "o2sim_par.root", bool isContinuous = true, unsigned threads = 0);

void run_clus_tpc(Int_t nEvents, TString mcEngine = "TGeant3", bool isContinuous = true, unsigned threads = 0)
{
  // Input and output file name
  std::stringstream inputfile, outputfile, paramfile;
  inputfile << "AliceO2_" << mcEngine << ".tpc.digi_" << nEvents << "_event.root";
  paramfile << "AliceO2_" << mcEngine << ".tpc.params_" << nEvents << ".root";
  outputfile << "AliceO2_" << mcEngine << ".tpc.clusters_" << nEvents << "_event.root";
  run_clus_tpc(outputfile.str(), inputfile.str(), paramfile.str(), isContinuous, threads);
}

void run_clus_tpc(std::string outputfile, std::string inputfile, std::string paramfile, bool isContinuous,
                  unsigned threads)
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("HIGH");
  logger->SetLogScreenLevel("DEBUG");

  // Setup timer
  TStopwatch timer;

  // Setup FairRoot analysis manager
  FairRunAna* run = new FairRunAna();
  FairFileSource* fFileSource = new FairFileSource(inputfile.data());
  run->SetSource(fFileSource);
  run->SetOutputFile(outputfile.data());

  // Setup Runtime DB
  FairRuntimeDb* rtdb = run->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.data());
  rtdb->setFirstInput(parInput1);

  TGeoManager::Import("geofile_full.root");

  // Setup clusterer
  o2::tpc::ClustererTask* clustTPC = new o2::tpc::ClustererTask(0);
  clustTPC->setContinuousReadout(isContinuous);

  run->AddTask(clustTPC);

  // Initialize everything
  run->Init();

  // Start simulation
  timer.Start();
  run->Run();

  run->TerminateRun();
  // we are done, cleanup
  delete clustTPC;

  std::cout << std::endl
            << std::endl;

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

  std::cout << std::endl
            << std::endl;
  std::cout << "Output file is " << outputfile.data() << std::endl;
  // std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << std::endl
            << std::endl;
  std::cout << "Macro finished succesfully." << std::endl;
  return;
}
