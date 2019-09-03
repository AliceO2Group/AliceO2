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
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>
#include <TParameter.h>
#include <string>
#include <FairLogger.h>
#include <TStopwatch.h>

#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

#include "GlobalTracking/CalibTOF.h"
#endif

#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "err.h"

void run_calib_tof(std::string path = "./", std::string outputfile = "o2calparams_tof.root",
                   std::string inputfileCalib = "o2calibration_tof.root")
{
  bool onlymerge = false; // set to true if you have already the outputs from forked processes and you want only merge

  TString namefile(outputfile);
  namefile.ReplaceAll(".root", "");

  const int ninstance = 4;
  o2::globaltracking::CalibTOF calib;
  calib.setDebugMode(1);

  if (path.back() != '/') {
    path += '/';
  }

  //>>>---------- attach input data --------------->>>
  TChain tofCalibInfo("calibrationTOF");
  tofCalibInfo.AddFile((path + inputfileCalib).data());
  TFile* fin = TFile::Open((path + inputfileCalib).data());
  TParameter<int>* minTimestamp = (TParameter<int>*)fin->Get("minTimestamp");
  TParameter<int>* maxTimestamp = (TParameter<int>*)fin->Get("maxTimestamp");

  calib.setInputTreeTOFCollectedCalibInfo(&tofCalibInfo);
  calib.setMinTimestamp(minTimestamp->GetVal());
  calib.setMaxTimestamp(maxTimestamp->GetVal());
  fin->Close();

  //<<<---------- attach input data ---------------<<<

  //-------------------- settings -----------//
  //calib.run(o2::globaltracking::CalibTOF::kLHCphase);
  //calib.run(o2::globaltracking::CalibTOF::kChannelOffset);
  //calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing); // all sectors

  // to be generalized for more than 2 forks
  int counter = 0;

  pid_t pids[ninstance];
  int n = ninstance;
  /* Start children. */
  TStopwatch timerTot;
  timerTot.Start();
  if (!onlymerge) {
    for (int i = 0; i < n; ++i) {
      if ((pids[i] = fork()) < 0) {
        perror("fork");
        abort();
      } else if (pids[i] == 0) {
        cout << "child " << i << endl;
        TFile outFile((path + namefile.Data() + Form("_fork%i.root", i)).data(), "recreate");
        TTree outTree("calibrationTOF", "Calibration TOF params");
        calib.setOutputTree(&outTree);
        calib.init();
        cout << i << ") Child process: My process id = " << getpid() << endl;
        cout << "Child process: Value returned by fork() = " << pids[i] << endl;

        // only for the first child
        if (i == 0)
          calib.run(o2::globaltracking::CalibTOF::kLHCphase);
        for (int sect = i; sect < 18; sect += ninstance)
          //calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, sect);
          calib.run(o2::globaltracking::CalibTOF::kChannelOffset, sect);
        calib.fillOutput();
        outFile.cd();
        outTree.Write();
        if (i == 0)
          calib.getLHCphaseHisto()->Write();
        calib.getChTimeSlewingHistoAll()->Write();
        outFile.Close();
        exit(0);
      }
    }

    int status;
    pid_t pid;

    while (n > 0) {
      pid = wait(&status);
      printf("Child with PID %ld exited with status 0x%x.\n", (long)pid, status);
      --n; // TODO(pts): Remove pid from the pids array.
    }
  }

  timerTot.Stop();
  Printf("Time to run the calibration was:");
  timerTot.Print();
  printf("merge outputs\n");

  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("calibrationTOF", "Calibration TOF params");
  calib.setOutputTree(&outTree);
  calib.init();
  timerTot.Start(1);
  for (int i = 0; i < ninstance; ++i) {
    calib.merge((path + namefile.Data() + Form("_fork%i.root", i)).data());
  }
  timerTot.Stop();
  Printf("Time to run the merging was:");
  timerTot.Print();

  // flag problematics
  timerTot.Start(1);
  calib.flagProblematics();
  timerTot.Stop();
  Printf("Time to run for problematics:");
  timerTot.Print();

  outFile.cd();
  calib.fillOutput();
  outTree.Write();
  outFile.Close();
}
