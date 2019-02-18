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

  int ninstance = 2;
  o2::globaltracking::CalibTOF calib[2];
  for(int i=0; i < ninstance; i++)
    calib[i].setDebugMode(1);

  if (path.back() != '/') {
    path += '/';
  }

  //>>>---------- attach input data --------------->>>
  TChain tofCalibInfo("calibrationTOF");
  tofCalibInfo.AddFile((path + inputfileCalib).data());
  TFile* fin = TFile::Open((path + inputfileCalib).data());
  TParameter<int>* minTimestamp = (TParameter<int>*)fin->Get("minTimestamp");
  TParameter<int>* maxTimestamp = (TParameter<int>*)fin->Get("maxTimestamp");

  for(int i=0; i < ninstance; i++){
    calib[i].setInputTreeTOFCollectedCalibInfo(&tofCalibInfo);
    calib[i].setMinTimestamp(minTimestamp->GetVal());
    calib[i].setMaxTimestamp(maxTimestamp->GetVal());
  }
  fin->Close();
  
  //<<<---------- attach input data ---------------<<<

  // create/attach output tree
  TFile outFile((path + outputfile).data(), "recreate");
  TTree outTree("calibrationTOF", "Calibration TOF params");

  for(int i=0; i < ninstance; i++)
    calib[i].setOutputTree(&outTree);
  
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

  for(int i=0; i < ninstance; i++)
    calib[i].init();

  //  calib.run(o2::globaltracking::CalibTOF::kLHCphase);
  //calib.run(o2::globaltracking::CalibTOF::kChannelOffset);
  //calib.run(o2::globaltracking::CalibTOF::kChannelTimeSlewing); // all sectors

  // to be generalized for more than 2 forks
  int counter = 0;
  pid_t pid = fork();

  if (pid == 0){ // child process
    printf("strip fork 1\n");
    calib[0].run(o2::globaltracking::CalibTOF::kLHCphase);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 0);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 1);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 2);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 3);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 4);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 5);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 6);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 7);
    calib[0].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 8);
    }
  else if (pid > 0){ //parent process
    printf("strip fork 2\n");
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 9);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 10);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 11);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 12);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 13);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 14);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 15);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 16);
    calib[1].run(o2::globaltracking::CalibTOF::kChannelTimeSlewing, 17);
  }

  for(int i=1; i < ninstance; i++)
     calib[0] += calib[i];

  calib[0].fillOutput();


  outFile.cd();
  outTree.Write();
  calib[0].getLHCphaseHisto()->Write();
  calib[0].getChTimeSlewingHistoAll()->Write();
  outFile.Close();

}
