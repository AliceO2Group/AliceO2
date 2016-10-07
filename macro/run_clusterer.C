#if !defined(__CINT__) || defined(__MAKECINT__)
  #include <Rtypes.h>
  #include <TString.h>
  #include <TStopwatch.h>
  #include <TGeoManager.h>

  #include "FairLogger.h"
  #include "FairRunAna.h"
  #include "FairFileSource.h"
  #include "FairSystemInfo.h"
  #include "FairRuntimeDb.h"
  #include "FairParRootFileIo.h"

  #include "TPCSimulation/ClustererTask.h"
#endif

void run_clusterer(Int_t nEvents = 2, TString mcEngine = "TGeant3")
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
  FairRunAna * fRun = new FairRunAna;
  FairFileSource *fFileSource = new FairFileSource(inputfile.str().c_str());
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(outputfile.str().c_str());

  // Setup Runtime DB
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.str().c_str());
  rtdb->setFirstInput(parInput1);

  TGeoManager::Import("geofile_full.root");

  // Setup clusterer
  AliceO2::TPC::ClustererTask *clustTPC = new AliceO2::TPC::ClustererTask;
  clustTPC->setClustererEnable(AliceO2::TPC::ClustererTask::ClustererType::Box,true);
  clustTPC->setClustererEnable(AliceO2::TPC::ClustererTask::ClustererType::HW,true);

  fRun->AddTask(clustTPC);

  // Initialize everything
  fRun->Init();

  clustTPC->getHwClusterer()->setProcessingType(AliceO2::TPC::HwClusterer::Processing::Parallel);
  //clustTPC->getHwClusterer()->setProcessingType(AliceO2::TPC::HwClusterer::Processing::Sequential);

  // Start simulation
  timer.Start();
  fRun->Run();

  fRun->TerminateRun();
  // we are done, cleanup
  delete clustTPC;


  std::cout << std::endl << std::endl;

  // Extract the maximal used memory an add is as Dart measurement
  // This line is filtered by CTest and the value send to CDash
  FairSystemInfo sysInfo;
  Float_t maxMemory=sysInfo.GetMaxMemory();
  std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage=ctime/rtime;
  std::cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
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
