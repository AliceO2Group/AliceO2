/// \file Digitizer.cxx
/// \brief Implementation of the ALICE TPC digitizer task
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TFile.h"
#include "TTree.h"

#include "TPCSimulation/DigitizerTask.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Constants.h"
#include "TPCSimulation/Point.h"

#include "FairLogger.h"
#include "FairRootManager.h"

//#include "valgrind/callgrind.h"

ClassImp(o2::TPC::DigitizerTask)

using namespace o2::TPC;


DigitizerTask::DigitizerTask()
  : FairTask("TPCDigitizerTask")
  , mDigitizer(nullptr)
  , mDigitContainer(nullptr)
  , mPointsArray(nullptr)
  , mDigitsArray(nullptr)
  , mHitFileName()
{
  /// \todo get rid of new
  mDigitizer = new Digitizer;
  //CALLGRIND_START_INSTRUMENTATION;
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

  //CALLGRIND_STOP_INSTRUMENTATION;
  //CALLGRIND_DUMP_STATS;
}


InitStatus DigitizerTask::Init()
{
  /// Initialize the task and the input and output containers
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
  mDigitsArray = new TClonesArray("o2::TPC::DigitMC");
  mgr->Register("TPCDigitMC", "TPC", mDigitsArray, kTRUE);
  
  mDigitizer->init();
  mDigitContainer = mDigitizer->getDigitContainer();
  return kSUCCESS;
}

void DigitizerTask::Exec(Option_t *option)
{
  FairRootManager *mgr = FairRootManager::Instance();

  /// Execute the digitization
  if (mHitFileName.size()) fillHitArrayFromFile();

  const int eventTime = Digitizer::getTimeBinFromTime(mgr->GetEventTime() * 0.001);

  LOG(DEBUG) << "Running digitization on new event at time bin " << eventTime << FairLogger::endl;
  mDigitsArray->Delete();
  mDigitContainer = mDigitizer->Process(mPointsArray);
  mDigitContainer->fillOutputContainer(mDigitsArray, eventTime);
  mgr->Fill();
//  std::vector<CommonMode> commonModeContainer(0);
//  digits->processCommonMode(commonModeContainer);
//  digits->fillOutputContainer(mDigitsArray, commonModeContainer);
}

void DigitizerTask::FinishTask()
{
  FairRootManager *mgr = FairRootManager::Instance();
  mDigitsArray->Delete();
  mDigitContainer->fillOutputContainer(mDigitsArray, 1000000);
  mgr->Fill();
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
    mPointsArray = new TClonesArray("o2::TPC::Point");
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
    new(dummy[size]) Point(fX, fY, fZ, fTime, fQ, fTrack, 98);
  }

  printf("Converted hits: %d\n", dummy.GetEntriesFast());
  delete tIn;
  ++eventNumber;
}
