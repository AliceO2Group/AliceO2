#if (!defined(__CINT__) && !defined(__CLING__)) || defined(__MAKECINT__)
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

  #include "TPCSimulation/ClustererTask.h"
#endif

void run_clus_tpc(Int_t nEvents = 10, TString mcEngine = "TGeant3")
{
  // Initialize logger
  FairLogger *logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("HIGH");
  logger->SetLogScreenLevel("DEBUG");

  // Input and output file name
  std::stringstream inputfile, outputfile, paramfile;
  inputfile << "AliceO2_" << mcEngine << ".tpc.digi_" << nEvents << "_event.root";
  paramfile << "AliceO2_" << mcEngine << ".tpc.params_" << nEvents << ".root";
  outputfile << "AliceO2_" << mcEngine << ".tpc.clusters_" << nEvents << "_event.root";

  // Setup timer
  TStopwatch timer;

  // Setup FairRoot analysis manager
  FairRunAna * run = new FairRunAna;
  FairFileSource *fFileSource = new FairFileSource(inputfile.str().c_str());
  run->SetSource(fFileSource);
  run->SetOutputFile(outputfile.str().c_str());

  // Setup Runtime DB
  FairRuntimeDb* rtdb = run->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.str().c_str());
  rtdb->setFirstInput(parInput1);

  TGeoManager::Import("geofile_full.root");

  // Setup clusterer
  o2::TPC::ClustererTask *clustTPC = new o2::TPC::ClustererTask;
  clustTPC->setClustererEnable(o2::TPC::ClustererTask::ClustererType::Box,false);
  clustTPC->setClustererEnable(o2::TPC::ClustererTask::ClustererType::HW,true);

  run->AddTask(clustTPC);

  // Initialize everything
  run->Init();

//  clustTPC->getHwClusterer()->setProcessingType(o2::TPC::HwClusterer::Processing::Parallel);
  clustTPC->getHwClusterer()->setProcessingType(o2::TPC::HwClusterer::Processing::Sequential);

  // Start simulation
  timer.Start();
  run->Run();

  run->TerminateRun();
  // we are done, cleanup
  delete clustTPC;


  std::cout << std::endl << std::endl;

  // Extract the maximal used memory an add is as Dart measurement
  // This line is filtered by CTest and the value send to CDash
  FairSystemInfo sysInfo;
  Float_t maxMemory=sysInfo.GetMaxMemory();
  std::cout << R"(<DartMeasurement name="MaxMemory" type="numeric/double">)";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage=ctime/rtime;
  std::cout << R"(<DartMeasurement name="CpuLoad" type="numeric/double">)";
  std::cout << cpuUsage;
  std::cout << "</DartMeasurement>" << std::endl;

  std::cout << std::endl << std::endl;
  std::cout << "Output file is "    << outputfile.str() << std::endl;
  //std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime
	    << "s" << std::endl << std::endl;
  std::cout << "Macro finished succesfully." << std::endl;
  return;

}
