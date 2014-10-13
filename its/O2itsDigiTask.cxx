#include "O2itsDigiTask.h"

#include "FairLink.h"
#include "FairRootManager.h"

#include "O2itsDigi.h"
#include "O2itsPoint.h"

#include "Riosfwd.h"                    // for ostream
#include "TClonesArray.h"               // for TClonesArray
#include "TMath.h"                      // for Sqrt
#include "TRandom.h"                    // for TRandom, gRandom

#include <stddef.h>                     // for NULL
#include <iostream>                     // for operator<<, basic_ostream, etc

O2itsDigiTask::O2itsDigiTask():
  FairTask("O2itsDigiTask"),
  fTimeResolution(0.),
  fPointArray(NULL),
  fDigiArray(NULL)
{
}

O2itsDigiTask::~O2itsDigiTask()
{
}

InitStatus O2itsDigiTask::Init()
{
  FairRootManager* ioman = FairRootManager::Instance();
  if (!ioman) {
    std::cout << "-E- O2itsDigiTask::Init: " ///todo replace with logger!
              << "RootManager not instantiated!" << std::endl;
    return kFATAL;
  }

  fPointArray = (TClonesArray*) ioman->GetObject("O2itsPoint");
  if (!fPointArray) {
    std::cout << "-W- O2itsDigiTask::Init: "
              << "No Point array!" << std::endl;
    return kERROR;
  }

  // Create and register output array
  fDigiArray = new TClonesArray("O2itsDigi");
  ioman->Register("O2itsDigi", "O2its", fDigiArray, kTRUE);

  return kSUCCESS;
}

void O2itsDigiTask::Exec(Option_t* opt)
{

  fDigiArray->Delete();

  // fill the map
  for(int ipnt = 0; ipnt < fPointArray->GetEntries(); ipnt++) {
    O2itsPoint* point = (O2itsPoint*) fPointArray->At(ipnt);
    if(!point) { continue; }

    Int_t xPad = CalcPad(point->GetX(), point->GetXOut());
    Int_t yPad = CalcPad(point->GetY(), point->GetYOut());
    Int_t zPad = CalcPad(point->GetZ(), point->GetZOut());

    Double_t timestamp = CalcTimeStamp(point->GetTime());

    O2itsDigi* digi = new ((*fDigiArray)[ipnt]) O2itsDigi(xPad, yPad, zPad, timestamp);
    if (fTimeResolution > 0) {
      digi->SetTimeStampError(fTimeResolution/TMath::Sqrt(fTimeResolution));
    } else {
      digi->SetTimeStampError(0);
    }

    digi->SetLink(FairLink("O2itsPoint", ipnt));
  }
}

Int_t O2itsDigiTask::CalcPad(Double_t posIn, Double_t posOut)
{
  Int_t result = (Int_t)(posIn + posOut)/2;
  return result;
}

Double_t O2itsDigiTask::CalcTimeStamp(Double_t timeOfFlight)
{
  Double_t eventTime = FairRootManager::Instance()->GetEventTime();
  Double_t detectionTime = gRandom->Gaus(0, fTimeResolution);

  Double_t result = eventTime + timeOfFlight + detectionTime;

  if (result < 0) {
    return 0;
  } else {
    return result;
  }
}

ClassImp(O2itsDigiTask)
