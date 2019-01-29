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

  o2::globaltracking::CollectCalibInfoTOF calib;
  // calib.setDebugFlag(1,1); // not implementented

  if (path.back() != '/') {
    path += '/';
  }

  //>>>---------- attach input data --------------->>>
  TChain tofCalibInfo("calibTOF");
  tofCalibInfo.AddFile((path + inputfileCalib).data());
  calib.setInputTreeTOFCalibInfo(&tofCalibInfo);

  //<<<---------- attach input data ---------------<<<

  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("calibrationTOF", "Calibration TOF params");
  calib.setOutputTree(&outTree);

  calib.init();

  calib.run();

  outFile.cd();
  outTree.Write();

}
