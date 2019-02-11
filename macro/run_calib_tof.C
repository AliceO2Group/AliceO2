#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TGeoGlobalMagField.h>
#include <TParameter.h>
#include <string>
#include <FairLogger.h>

#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"

#include <unistd.h>

#include "GlobalTracking/CalibTOF.h"
#endif

void run_calib_tof(std::string path = "./", std::string outputfile = "o2calparams_tof.root",
		   std::string inputfileCalib = "o2calibration_tof.root")
{

  o2::globaltracking::CalibTOF calib;
  // calib.setDebugFlag(1,1); // not implementented

  if (path.back() != '/') {
    path += '/';
  }

  //>>>---------- attach input data --------------->>>
  TChain tofCalibInfo("calibrationTOF");
  tofCalibInfo.AddFile((path + inputfileCalib).data());
  calib.setInputTreeTOFCollectedCalibInfo(&tofCalibInfo);
  TFile* fin = TFile::Open((path + inputfileCalib).data());
  TParameter<int>* minTimestamp = (TParameter<int>*)fin->Get("minTimestamp");
  TParameter<int>* maxTimestamp = (TParameter<int>*)fin->Get("maxTimestamp");
  calib.setMinTimestamp(minTimestamp->GetVal());
  calib.setMaxTimestamp(maxTimestamp->GetVal());
  fin->Close();
  
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

  //  calib.run(o2::globaltracking::CalibTOF::kLHCphase);
  //calib.run(o2::globaltracking::CalibTOF::kChannelOffset);
  //calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing); // all sectors

  int counter = 0;
  pid_t pid = fork();

  if (pid == 0){ // child process
    printf("strip fork 1\n");
    calib.run(o2::globaltracking::CalibTOF::kLHCphase);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 0);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 1);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 2);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 3);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 4);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 5);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 6);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 7);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 8);
    }
  else if (pid > 0){ //parent process
    printf("strip fork 2\n");
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 9);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 10);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 11);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 12);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 13);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 14);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 15);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 16);
    calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 17);
  }

  calib.fillOutput();


  outFile.cd();
  outTree.Write();
  calib.getLHCphaseHisto()->Write();
  calib.getChTimeSlewingHistoAll()->Write();

  outFile.Close();

}
