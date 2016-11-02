/// \file FindTracks.h
/// \brief Simple track finding from the hits
/// \author bogdan.vulpescu@cern.ch 
/// \date 07/10/2016

#include "MFTReconstruction/Hit.h"
#include "MFTReconstruction/Track.h"
#include "MFTReconstruction/FindTracks.h"

#include "TClonesArray.h"
#include "TF1.h"
#include "TGraphErrors.h"

#include "FairLogger.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::FindTracks)

//_____________________________________________________________________________
FindTracks::FindTracks():
fHits(NULL),
fTracks(NULL),
fNHits(0),
fNTracks(0),
fTNofEvents(0),
fTNofHits(0),
fTNofTracks(0)
{

}

//_____________________________________________________________________________
FindTracks::FindTracks(Int_t iVerbose)
  : FairTask("MFT Track Finder", iVerbose),
    fHits(NULL),
    fTracks(NULL),
    fNHits(0),
    fNTracks(0),
    fTNofEvents(0),
    fTNofHits(0),
    fTNofTracks(0)
{

}

//_____________________________________________________________________________
FindTracks::~FindTracks()
{

  Reset();
  if (fTracks) {
    fTracks->Delete();
    delete fTracks;
  }

}

//_____________________________________________________________________________
InitStatus FindTracks::Init()
{

  LOG(INFO) << "Initilization of FindTracks"
            << FairLogger::endl;

  // Get a handle from the IO manager
  FairRootManager* ioman = FairRootManager::Instance();
  if ( !ioman ) {
    LOG(FATAL) << "RootManager not instantiated!" << FairLogger::endl;
    return kFATAL;
  }

  // Get a pointer to the previous already existing data level
  fHits = static_cast<TClonesArray*>(ioman->GetObject("MFTHits"));
  if ( ! fHits ) {
    LOG(ERROR) << "No InputDataLevelName array! "
               << "FindTracks will be inactive" << FairLogger::endl;
    return kERROR;
  }

  fTracks = new TClonesArray("AliceO2::MFT::Track", 100);
  ioman->Register("MFTTracks","MFT",fTracks,kTRUE);

  // Do whatever else is needed at the initilization stage
  // Create histograms to be filled
  // initialize variables

  return kSUCCESS;

}

//_____________________________________________________________________________
InitStatus FindTracks::ReInit()
{

  LOG(DEBUG) << "Re-Initilization of FindTracks"
             << FairLogger::endl;

  return kSUCCESS;

}

//_____________________________________________________________________________
void FindTracks::InitMQ(TList* tempList) 
{

  LOG(INFO) << "FindTracks::InitMQ >>>>>" << "";

  fTracks = new TClonesArray("AliceO2::MFT::Track",10000);

  return;

}

//_____________________________________________________________________________
void FindTracks::Exec(Option_t* /*opt*/) 
{

  Info("Exec","Exec called",0,0);

  Reset();

  const Int_t nMaxTracks = 100;
  fNHits = fHits->GetEntriesFast();

  Float_t **xPos = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) xPos[i] = new Float_t[fNHits];
  Float_t **yPos = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) yPos[i] = new Float_t[fNHits];
  Float_t **zPos = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) zPos[i] = new Float_t[fNHits];
  Float_t **xPosErr = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) xPosErr[i] = new Float_t[fNHits];
  Float_t **yPosErr = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) yPosErr[i] = new Float_t[fNHits];

  Int_t nTrackHits[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) nTrackHits[i] = 0;

  AliceO2::MFT::Hit *hit;
  Int_t iMCindex;
  for (Int_t iHit = 0; iHit < fNHits; iHit++) {
    hit = static_cast<AliceO2::MFT::Hit*>(fHits->At(iHit));
    if ( !hit) { continue; }

    iMCindex = hit->GetRefIndex();

    xPos[iMCindex][nTrackHits[iMCindex]] = hit->GetX();
    yPos[iMCindex][nTrackHits[iMCindex]] = hit->GetY();
    zPos[iMCindex][nTrackHits[iMCindex]] = hit->GetZ();

    xPosErr[iMCindex][nTrackHits[iMCindex]] = hit->GetDx();
    yPosErr[iMCindex][nTrackHits[iMCindex]] = hit->GetDy();

    //printf("Hits %6d: %10.4f %10.4f %10.4f %10.4f %10.4f %d %d \n",iHit,hit->GetX(),hit->GetY(),hit->GetZ(),hit->GetDx(),hit->GetDy(),iMCindex,nTrackHits[iMCindex]);

    nTrackHits[iMCindex]++;

  }

  // a simple fit through the hits
  
  TF1* f1 = new TF1("f1", "[0]*x + [1]");
  TGraphErrors *trackXZ, *trackYZ;

  Double_t slopeX, offX, chi2X, slopeY, offY, chi2Y;
  for (Int_t iTrack = 0; iTrack < nMaxTracks; iTrack++) {

    if (nTrackHits[iTrack] < 4) continue; 

    trackXZ = new TGraphErrors(nTrackHits[iTrack], zPos[iTrack], xPos[iTrack], 0, xPosErr[iTrack]);
    trackXZ->Fit("f1", "Q");
    slopeX = f1->GetParameter(0);
    offX = f1->GetParameter(1);
    chi2X = f1->GetChisquare();

    trackYZ = new TGraphErrors(nTrackHits[iTrack], zPos[iTrack], yPos[iTrack], 0, yPosErr[iTrack]);
    trackYZ->Fit("f1", "Q");
    slopeY = f1->GetParameter(0);
    offY = f1->GetParameter(1);
    chi2Y = f1->GetChisquare();

    Track* track = new Track();
    track->SetX(offX);
    track->SetY(offY);
    track->SetZ(0.);
    track->SetTx(slopeX);
    track->SetTy(slopeY);
    new ((*fTracks)[fNTracks]) Track(*track);
    fNTracks++;
    delete track;

  }  

  delete [] xPos;
  delete [] yPos;
  delete [] zPos;
  delete [] xPosErr;
  delete [] yPosErr;

  delete trackXZ;
  delete trackYZ;

  fTNofEvents++;
  fTNofHits   += fNHits;
  fTNofTracks += fNTracks;

}

//_____________________________________________________________________________
void FindTracks::ExecMQ(TList* inputList,TList* outputList) 
{

  LOG(INFO) << "FindTracks::ExecMQ >>>>> (" << inputList->GetName() << "," << outputList->GetName() << "), Event " << fTNofEvents << "";

  fHits = (TClonesArray*) inputList->FindObject("MFTHits");
  outputList->Add(fTracks);

  Exec("");

  return;

}

//_____________________________________________________________________________
void FindTracks::Reset() 
{

  fNTracks = fNHits = 0;
  if ( fTracks ) fTracks->Clear();

}
