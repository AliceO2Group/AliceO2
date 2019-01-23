#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>
#include <string>
#include <FairLogger.h>

#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

#include "GlobalTracking/CalibTOF.h"
#endif

void run_calib_tof(std::string path = "./", std::string outputfile = "o2calibration_tof.root",
		   std::string inputfileCalib = "o2calib_tof.root")
{

  o2::globaltracking::CalibTOF calib;
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

  //-------------------- settings -----------//
  /*
    matching.setITSROFrameLengthMUS(5.0f); // ITS ROFrame duration in \mus
    matching.setCutMatchingChi2(100.);
    std::array<float, o2::track::kNParams> cutsAbs = { 2.f, 2.f, 0.2f, 0.2f, 4.f };
    std::array<float, o2::track::kNParams> cutsNSig2 = { 49.f, 49.f, 49.f, 49.f, 49.f };
    matching.setCrudeAbsDiffCut(cutsAbs);
    matching.setCrudeNSigma2Cut(cutsNSig2);
    matching.setTPCTimeEdgeZSafeMargin(3);
  */
  calib.init();

  calib.run();

  outFile.cd();
  outTree.Write();
  outFile.Close();

}
