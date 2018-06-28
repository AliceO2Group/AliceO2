#if !defined(__CLING__) || defined(__ROOTCLING__)
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

#include "FITSimulation/DigitizerTask.h"
#include "FITReconstruction/CollisionTimeRecoTask.h"

#endif

void run_reco_fit(float rate= 50.e3/*,
				     std::string inputGRP = "o2sim_grp.root"*/) {
  gSystem->Load("libFITBase");
  gSystem->Load("libFITSimulation");
  gSystem->Load("libFITReconstruction");
  // Initialize logger
  FairLogger *logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("DEBUG");
    
  // Setup timer
  TStopwatch timer;
  std::stringstream inputfile, outputfile, paramfile;
  inputfile << "o2sim_digi.root";
  paramfile << "o2sim_par.root";
  outputfile << "o2reco_fit.root";

  // Setup FairRoot analysis manager
  FairRunAna * run = new FairRunAna();
  FairFileSource *fFileSource = new FairFileSource(inputfile.str().c_str());

  run->SetSource(fFileSource);
  run->SetOutputFile(outputfile.str().c_str());
  if (rate > 0) {
    fFileSource->SetEventMeanTime(1.e9 / rate); // is in us
    std::cout<<"@@@@ rate > 0"<<std::endl;
  }
 
  // Setup Runtime DB
  FairRuntimeDb* rtdb = run->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.str().c_str());
  rtdb->setFirstInput(parInput1);
  
   
  o2::fit::CollisionTimeRecoTask *recoFIT = new o2::fit::CollisionTimeRecoTask();
  run->AddTask(recoFIT);
  run->Init();

  timer.Start();
  run->Run();
  
  
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
  std::cout << "Macro finished succesfully." << std::endl;
  
  std::cout << std::endl << std::endl;
  std::cout << "Output file is "    << outputfile.str().c_str() << std::endl;
  //std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime
	    << "s" << std::endl << std::endl;
  
}
