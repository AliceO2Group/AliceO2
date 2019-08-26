// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include <TStopwatch.h>

#include "FairLogger.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"

#include "TOFReconstruction/ClustererTask.h"
#endif

void run_clus_tof(std::string outputfile = "o2clus_tof.root", std::string inputfile = "o2sim_digi.root", std::string paramfile = "o2sim_par.root")
{
  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("DEBUG");

  // Setup timer
  TStopwatch timer;

  // Setup FairRoot analysis manager
  FairRunAna* fRun = new FairRunAna();
  FairFileSource* fFileSource = new FairFileSource(inputfile.data());
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(outputfile.data());

  // Setup Runtime DB
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.data());
  rtdb->setFirstInput(parInput1);

  // Setup clusterizer
  Bool_t useMCTruth = kTRUE; // kFALSE if no comparison with MC needed
  o2::tof::ClustererTask* clus = new o2::tof::ClustererTask(useMCTruth);

  fRun->AddTask(clus);

  fRun->Init();

  timer.Start();
  fRun->Run();

  std::cout << std::endl
            << std::endl;

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
  cout << "</DartMeasurement>" << endl;
  cout << endl
       << endl;
  std::cout << "Macro finished succesfully." << std::endl;
  std::cout << "Output file is " << outputfile.data() << std::endl;
  // std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl
            << endl;
}
