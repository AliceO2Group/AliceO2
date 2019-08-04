#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <sstream>

#include <TStopwatch.h>
#include "DataFormatsParameters/GRPObject.h"
#include "FairLogger.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairRuntimeDb.h"
#include "FairParRootFileIo.h"
#include "FairSystemInfo.h"

#include "MFTSimulation/DigitizerTask.h"
#endif

int updateMFTinGRP(std::string inputGRP, std::string grpName = "GRP");

void run_digi_mft(float rate = 50e3,
                  std::string outputfile = "o2digi_mft.root",
                  std::string inputfile = "o2sim.root",
                  std::string paramfile = "o2sim_par.root",
                  std::string inputGRP = "o2sim_grp.root",
                  std::string inputfileQED = "", // "o2sim_QED.root" // optional QED hits file
                  float timebinQEDns = 1000)     // each entry of QED hits file corresponds to as many nanoseconds
{
  // if rate>0 then continuous simulation for this rate will be performed

  // Initialize logger
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  // Setup timer
  TStopwatch timer;

  // Setup FairRoot analysis manager
  FairRunAna* fRun = new FairRunAna();
  FairFileSource* fFileSource = new FairFileSource(inputfile);
  fRun->SetSource(fFileSource);
  fRun->SetOutputFile(outputfile.data());

  if (rate > 0) {
    fFileSource->SetEventMeanTime(1.e9 / rate); // is in us
    updateMFTinGRP(inputGRP);
  }

  // Setup Runtime DB
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  FairParRootFileIo* parInput1 = new FairParRootFileIo();
  parInput1->open(paramfile.data());
  rtdb->setFirstInput(parInput1);

  // Setup digitizer
  // Call o2::mft::DigitizerTask(kTRUE) to activate the ALPIDE simulation
  o2::mft::DigitizerTask* digi = new o2::mft::DigitizerTask();
  //
  // This is an example of setting the digitization parameters manually
  // ====>>
  // defaults
  digi->getDigiParams().setContinuous(rate > 0);     // continuous vs per-event mode
  digi->getDigiParams().setROFrameLength(10000);     // RO frame in ns
  digi->getDigiParams().setStrobeDelay(0);           // Strobe delay wrt beginning of the RO frame, in ns
  digi->getDigiParams().setStrobeLength(10000 - 25); // Strobe length in ns
  // parameters of signal time response: flat-top duration, max rise time and q @ which rise time is 0
  digi->getDigiParams().getSignalShape().setParameters(7500., 1100., 450.);
  digi->getDigiParams().setChargeThreshold(150); // charge threshold in electrons
  //digi->getDigiParams().setNoisePerPixel(1.e-6); // noise level
  digi->getDigiParams().setNoisePerPixel(0.); // noise level
  // <<===

  //-------------- do we have QED digits file provided ?
  TFile* qedHitsFile = nullptr;
  if (!inputfileQED.empty()) {
    qedHitsFile = TFile::Open(inputfileQED.data());
    if (!qedHitsFile || qedHitsFile->IsZombie()) {
      LOG(FATAL) << "Failed to open QED file " << inputfileQED << FairLogger::endl;
    }
    TTree* qedTree = (TTree*)qedHitsFile->Get("o2sim");
    if (!qedTree) {
      LOG(FATAL) << "No o2sim tree in QED file " << inputfileQED << FairLogger::endl;
    }
    TBranch* qedBranch = qedTree->GetBranch("MFTHit");
    if (!qedBranch) {
      LOG(FATAL) << "No MFTHit branch in the QED hits tree of file " << inputfileQED << FairLogger::endl;
    }
    digi->setQEDInput(qedBranch, timebinQEDns, 99); // the QED is assigned source ID = 99
  }

  digi->setFairTimeUnitInNS(1.0); // tell in which units (wrt nanosecond) FAIT timestamps are
  fRun->AddTask(digi);

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
  std::cout << "Macro finished succesfully" << std::endl;

  std::cout << endl
            << std::endl;
  std::cout << "Output file is " << outputfile << std::endl;
  // std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << "s" << endl
            << endl;

  if (qedHitsFile) {
    qedHitsFile->Close();
    delete qedHitsFile;
  }
}

int updateMFTinGRP(std::string inputGRP, std::string grpName)
{
  TFile flGRP(inputGRP.data(), "update");
  if (flGRP.IsZombie()) {
    LOG(ERROR) << "Failed to open in update mode " << inputGRP << FairLogger::endl;
    return -10;
  }
  auto grp =
    static_cast<o2::parameters::GRPObject*>(flGRP.GetObjectChecked(grpName.data(), o2::parameters::GRPObject::Class()));
  if (!grp) {
    LOG(ERROR) << "Did not find GRP object named " << inputGRP << FairLogger::endl;
    return -12;
  }

  vector<o2::detectors::DetID> contDet = {o2::detectors::DetID::MFT};

  for (auto det : contDet) {
    if (grp->isDetReadOut(det)) {
      grp->addDetContinuousReadOut(det);
    }
  }

  LOG(INFO) << "Updated GRP in " << inputGRP << " flagging continously read-out detector(s)" << FairLogger::endl;
  grp->print();
  flGRP.WriteObjectAny(grp, grp->Class(), grpName.data());
  flGRP.Close();
  return 0;
}
