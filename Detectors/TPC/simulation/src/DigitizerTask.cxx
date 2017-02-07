/// \file DigitizerTask.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TFile.h"
#include "TTree.h"

#include "TPCSimulation/DigitizerTask.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Constants.h"
#include "TPCSimulation/Point.h"

#include "FairLogger.h"
#include "FairRootManager.h"

#include "valgrind/callgrind.h"

ClassImp(AliceO2::TPC::DigitizerTask)

using namespace AliceO2::TPC;


DigitizerTask::DigitizerTask()
  : FairTask("TPCDigitizerTask"),
    mDigitizer(nullptr),
    mPointsArray(nullptr),
    mDigitsArray(nullptr),
    mHitFileName()
{
  /// @todo get rid of new
  mDigitizer = new Digitizer;
  CALLGRIND_START_INSTRUMENTATION;
}

DigitizerTask::~DigitizerTask()
{
  delete mDigitizer;
  if (mDigitsArray){
    delete mDigitsArray;
  }

  if (mHitFileName.c_str()) {
    mPointsArray->Delete();
    delete mPointsArray;
  }

  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS;
}


InitStatus DigitizerTask::Init()
{
  FairRootManager *mgr = FairRootManager::Instance();
  if(!mgr){
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }
  
  mPointsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("TPCPoint")); //TODO: does mPointsArray need to be deleted?
  if (!mPointsArray) {
    LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }
  
  // Register output container
  mDigitsArray = new TClonesArray("AliceO2::TPC::Digit");
  mgr->Register("TPCDigit", "TPC", mDigitsArray, kTRUE);
  
  mDigitizer->init();
  return kSUCCESS;
}

void DigitizerTask::Exec(Option_t *option)
{
  if (mHitFileName.size()) fillHitArrayFromFile();

  mDigitsArray->Delete();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
  
  DigitContainer *digits = mDigitizer->Process(mPointsArray);
  /// @todo: Digitizer.getDigitContainer()
  std::vector<CommonMode> commonModeContainer(0);
  digits->processCommonMode(commonModeContainer);
  digits->fillOutputContainer(mDigitsArray, commonModeContainer);

}

void DigitizerTask::fillHitArrayFromFile()
{
  static int eventNumber = 0;
  printf("Process Hits from %s event %d\n", mHitFileName.c_str(), eventNumber);

  TFile fIn(mHitFileName.c_str());
  TTree *tIn = static_cast<TTree*>(fIn.Get("tHit"));

  int   fEvent =0;
  float fQ     =0.f;
  float fTime  =0.f;
  int   fTrack =0.f;
  float fX     =0.f;
  float fY     =0.f;
  float fZ     =0.f;

  tIn->SetBranchAddress("fEvent",  &fEvent );
  tIn->SetBranchAddress("fQ",      &fQ     );
  tIn->SetBranchAddress("fTime",   &fTime  );
  tIn->SetBranchAddress("fTrack",  &fTrack );
  tIn->SetBranchAddress("fX",      &fX     );
  tIn->SetBranchAddress("fY",      &fY     );
  tIn->SetBranchAddress("fZ",      &fZ     );

  if (eventNumber==0) {
    mPointsArray = new TClonesArray("AliceO2::TPC::Point");
  }
  else {
    mPointsArray->Clear();
  }

  TClonesArray &dummy = *mPointsArray;


//   printf("%p: %d\n", tIn, tIn->GetEntries());
  for (int ihit=0; ihit<tIn->GetEntries(); ++ihit) {
//     printf("Processing hit %d (event %d)\n", ihit, fEvent);
    tIn->GetEntry(ihit);
    if (fEvent<eventNumber) continue;
    if (fEvent>eventNumber) break;
//     printf("Filling hit %d (event %d)\n", ihit, fEvent);

    const int size = dummy.GetEntriesFast();
    new(dummy[size]) Point(fTrack, 98, TVector3(fX, fY, fZ), TVector3(0,0,0),
                           fTime, 0., fQ*WION);
  }

  printf("Converted hits: %d\n", dummy.GetEntriesFast());
  delete tIn;
  ++eventNumber;
}
