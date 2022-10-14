// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>
#include <string>
#include <fairlogger/Logger.h>

#include "TOFCalibration/CollectCalibInfoTOF.h"
#endif

void run_collect_calib_tof(std::string path = "./", std::string outputfile = "o2calibration_tof.root",
                           std::string inputfileCalib = "o2calib_tof.root")
{

  o2::globaltracking::CollectCalibInfoTOF collect;
  // collect.setDebugFlag(1,1); // not implementented

  if (path.back() != '/') {
    path += '/';
  }

  //>>>---------- attach input data --------------->>>
  TChain tofCalibInfo("calibTOF");
  tofCalibInfo.AddFile((path + inputfileCalib).data());
  collect.setInputTreeTOFCalibInfo(&tofCalibInfo);

  //<<<---------- attach input data ---------------<<<

  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("calibrationTOF", "Calibration TOF params");
  collect.setOutputTree(&outTree);

  collect.init();

  collect.run();

  outFile.cd();
  outTree.Write();
  collect.getMinTimestamp().Write();
  collect.getMaxTimestamp().Write();
  outFile.Close();
}
