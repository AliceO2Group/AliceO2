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
#include <string>
#include <FairLogger.h>

#include "GlobalTracking/CollectCalibInfoTOF.h"
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
