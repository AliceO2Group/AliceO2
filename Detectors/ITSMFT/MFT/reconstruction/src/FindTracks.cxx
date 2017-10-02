// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FindTracks.h
/// \brief Simple track finding from the clusters
/// \author bogdan.vulpescu@cern.ch 
/// \date 07/10/2016

#include "ITSMFTReconstruction/Cluster.h"

#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/Track.h"
#include "MFTReconstruction/FindTracks.h"

#include "TClonesArray.h"
#include "TF1.h"
#include "TGraphErrors.h"

#include <FairMQLogger.h>

using namespace o2::MFT;

ClassImp(o2::MFT::FindTracks)

//_____________________________________________________________________________
FindTracks::FindTracks():
mClusters(nullptr),
mTracks(nullptr),
mNClusters(0),
mNTracks(0),
mTNofEvents(0),
mTNofClusters(0),
mTNofTracks(0),
mEventHeader(nullptr)
{

}

//_____________________________________________________________________________
FindTracks::FindTracks(Int_t iVerbose)
  : FairTask("MFT Track Finder", iVerbose),
    mClusters(nullptr),
    mTracks(nullptr),
    mNClusters(0),
    mNTracks(0),
    mTNofEvents(0),
    mTNofClusters(0),
    mTNofTracks(0),
    mEventHeader(nullptr)
{

}

//_____________________________________________________________________________
FindTracks::~FindTracks()
{

  reset();
  if (mTracks) {
    mTracks->Delete();
    delete mTracks;
  }

}

//_____________________________________________________________________________
InitStatus FindTracks::Init()
{

  LOG(INFO) << "FindTracks::Init >>>>> initilization" << "";

  // Get a handle from the IO manager
  FairRootManager* ioman = FairRootManager::Instance();
  if ( !ioman ) {
    LOG(INFO) << "FindTracks::Init >>>>> RootManager not instantiated!" << "";
    return kFATAL;
  }

  // Get a pointer to the previous already existing data level
  mClusters = static_cast<TClonesArray*>(ioman->GetObject("MFTClusters"));
  if ( ! mClusters ) {
    LOG(ERROR) << "FindTracks::Init >>>>> No InputDataLevelName array! "
               << "FindTracks will be inactive" << "";
    return kERROR;
  }

  mTracks = new TClonesArray("o2::MFT::Track", 100);
  ioman->Register("MFTTracks","MFT",mTracks,kTRUE);

  // Do whatever else is needed at the initilization stage
  // Create histograms to be filled
  // initialize variables

  return kSUCCESS;

}

//_____________________________________________________________________________
InitStatus FindTracks::ReInit()
{

  LOG(DEBUG) << "FindTracks::ReInit >>>>> Re-Initilization of FindTracks" << "";

  return kSUCCESS;

}

//_____________________________________________________________________________
void FindTracks::initMQ(TList* tempList) 
{

  LOG(INFO) << "FindTracks::initMQ >>>>>" << "";

  mTracks = new TClonesArray("o2::MFT::Track",10000);
  mTracks->Dump();

  return;

}

//_____________________________________________________________________________
void FindTracks::Exec(Option_t* /*opt*/) 
{

  //Info("Exec","Exec called",0,0);
  LOG(INFO) << "FindTracks::Exec >>>>>" << "";

  reset();
  /*
  const Int_t nMaxTracks = 1000;
  mNClusters = mClusters->GetEntriesFast();

  auto **xPos = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) xPos[i] = new Float_t[mNClusters];
  auto **yPos = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) yPos[i] = new Float_t[mNClusters];
  auto **zPos = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) zPos[i] = new Float_t[mNClusters];
  auto **xPosErr = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) xPosErr[i] = new Float_t[mNClusters];
  auto **yPosErr = new Float_t*[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) yPosErr[i] = new Float_t[mNClusters];

  Int_t nTrackClusters[nMaxTracks];
  for (Int_t i = 0; i < nMaxTracks; i++) nTrackClusters[i] = 0;
  
  o2::MFT::Cluster *cluster;
  Int_t iMCindex;
  for (Int_t iCls = 0; iCls < mNClusters; iCls++) {
    cluster = static_cast<o2::MFT::Cluster*>(mClusters->At(iCls));
    if ( !cluster) { continue; }

    iMCindex = cluster->GetRefIndex();

    xPos[iMCindex][nTrackClusters[iMCindex]] = cluster->GetX();
    yPos[iMCindex][nTrackClusters[iMCindex]] = cluster->GetY();
    zPos[iMCindex][nTrackClusters[iMCindex]] = cluster->GetZ();

    xPosErr[iMCindex][nTrackClusters[iMCindex]] = cluster->GetDx();
    yPosErr[iMCindex][nTrackClusters[iMCindex]] = cluster->GetDy();

    //printf("Clusters %6d: %10.4f %10.4f %10.4f %10.4f %10.4f %d %d \n",iCls,cluster->GetX(),cluster->GetY(),cluster->GetZ(),cluster->GetDx(),cluster->GetDy(),iMCindex,nTrackClusters[iMCindex]);

    nTrackClusters[iMCindex]++;

  }

  // a simple fit through the clusters
  
  auto* f1 = new TF1("f1", "[0]*x + [1]");
  TGraphErrors *trackXZ, *trackYZ;

  Double_t slopeX, offX, chi2X, slopeY, offY, chi2Y;
  for (Int_t iTrack = 0; iTrack < nMaxTracks; iTrack++) {

    if (nTrackClusters[iTrack] < 4) continue; 

    trackXZ = new TGraphErrors(nTrackClusters[iTrack], zPos[iTrack], xPos[iTrack], nullptr, xPosErr[iTrack]);
    trackXZ->Fit("f1", "Q");
    slopeX = f1->GetParameter(0);
    offX = f1->GetParameter(1);
    chi2X = f1->GetChisquare();

    trackYZ = new TGraphErrors(nTrackClusters[iTrack], zPos[iTrack], yPos[iTrack], nullptr, yPosErr[iTrack]);
    trackYZ->Fit("f1", "Q");
    slopeY = f1->GetParameter(0);
    offY = f1->GetParameter(1);
    chi2Y = f1->GetChisquare();

    auto* track = new Track();
    track->SetX(offX);
    track->SetY(offY);
    track->SetZ(0.);
    track->SetTx(slopeX);
    track->SetTy(slopeY);
    new ((*mTracks)[mNTracks]) Track(*track);
    mNTracks++;
    delete track;

  }  
  
  delete [] xPos;
  delete [] yPos;
  delete [] zPos;
  delete [] xPosErr;
  delete [] yPosErr;

  delete trackXZ;
  delete trackYZ;
  */
  mTNofEvents++;
  mTNofClusters   += mNClusters;
  mTNofTracks += mNTracks;

}

//_____________________________________________________________________________
void FindTracks::execMQ(TList* inputList,TList* outputList) 
{

  LOG(INFO) << "FindTracks::execMQ >>>>> add MFTClusters for event " << mTNofEvents << "";

  mClusters = (TClonesArray*)inputList->FindObject("MFTClusters");
  outputList->Add(mTracks);

  LOG(INFO) << "FindTracks::execMQ >>>>> add EventHeader. for event " << mTNofEvents << "";

  mEventHeader = (EventHeader*)inputList->FindObject("EventHeader.");
  outputList->Add(mEventHeader);

  Exec("");

  return;

}

//_____________________________________________________________________________
void FindTracks::reset() 
{

  mNTracks = mNClusters = 0;
  if ( mTracks ) mTracks->Clear();

}
