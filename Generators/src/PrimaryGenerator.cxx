// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - June 2017

#include "Generators/PrimaryGenerator.h"
#include "Generators/Generator.h"
#include "Generators/InteractionDiamondParam.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "FairLogger.h"

#include "FairGenericStack.h"
#include "TFile.h"
#include "TTree.h"

#include "TDatabasePDG.h"
#include "TVirtualMC.h"

using o2::dataformats::MCEventHeader;

namespace o2
{
namespace eventgen
{

/*****************************************************************/

PrimaryGenerator::~PrimaryGenerator()
{
  /** destructor **/

  if (mEmbedFile && mEmbedFile->IsOpen()) {
    mEmbedFile->Close();
    delete mEmbedFile;
  }
  if (mEmbedEvent)
    delete mEmbedEvent;
}

/*****************************************************************/

Bool_t PrimaryGenerator::Init()
{
  /** init **/

  LOG(INFO) << "Initialising primary generator";

  /** embedding **/
  if (mEmbedTree) {
    LOG(INFO) << "Embedding into: " << mEmbedFile->GetName()
              << " (" << mEmbedEntries << " events)";
    return FairPrimaryGenerator::Init();
  }

  /** normal generation **/

  /** retrieve and set interaction diamond **/
  auto& diamond = InteractionDiamondParam::Instance();
  setInteractionDiamond(diamond.position, diamond.width);

  /** base class init **/
  return FairPrimaryGenerator::Init();
}

/*****************************************************************/

Bool_t PrimaryGenerator::GenerateEvent(FairGenericStack* pStack)
{
  /** generate event **/

  /** normal generation if no embedding **/
  if (!mEmbedTree)
    return FairPrimaryGenerator::GenerateEvent(pStack);

  /** this is for embedding **/

  /** setup interaction vertex **/
  mEmbedTree->GetEntry(mEmbedIndex);
  setInteractionVertex(mEmbedEvent);

  /** notify event generators **/
  auto genList = GetListOfGenerators();
  for (int igen = 0; igen < genList->GetEntries(); ++igen) {
    auto o2gen = dynamic_cast<Generator*>(genList->At(igen));
    if (o2gen)
      o2gen->notifyEmbedding(mEmbedEvent);
  }

  /** generate event **/
  if (!FairPrimaryGenerator::GenerateEvent(pStack))
    return kFALSE;

  /** add embedding info to event header **/
  auto o2event = dynamic_cast<MCEventHeader*>(fEvent);
  if (o2event) {
    o2event->setEmbeddingFileName(mEmbedFile->GetName());
    o2event->setEmbeddingEventIndex(mEmbedIndex);
  }

  /** increment embedding counter **/
  mEmbedIndex++;
  mEmbedIndex %= mEmbedEntries;

  /** success **/
  return kTRUE;
}

/*****************************************************************/

void PrimaryGenerator::AddTrack(Int_t pdgid, Double_t px, Double_t py, Double_t pz,
                                Double_t vx, Double_t vy, Double_t vz,
                                Int_t mother1, Int_t mother2,
                                Int_t daughter1, Int_t daughter2,
                                Bool_t wanttracking,
                                Double_t e, Double_t tof,
                                Double_t weight, TMCProcess proc)
{
  /** add track **/

  /** add event vertex to track vertex **/
  vx += fVertex.X();
  vy += fVertex.Y();
  vz += fVertex.Z();

  /** check if particle to be tracked exists in PDG database **/
  auto particlePDG = TDatabasePDG::Instance()->GetParticle(pdgid);
  if (wanttracking && !particlePDG) {
    LOG(WARN) << "Particle to be tracked is not defined in PDG: pdg = " << pdgid;
    wanttracking = false;
  }

  /** set all other parameters required by PushTrack **/
  Int_t doTracking = 0; // Go to tracking
  if (fdoTracking && wanttracking) {
    doTracking = 1;
  }
  Int_t dummyparent = -1; // Primary particle (now the value is -1 by default)
  Double_t polx = 0.;     // Polarisation
  Double_t poly = 0.;
  Double_t polz = 0.;
  Int_t ntr = 0;    // Track number; to be filled by the stack
  Int_t status = 0; // Generation status

  // correct for tracks which are in list before generator is called
  if (mother1 != -1) {
    mother1 += fMCIndexOffset;
  }
  if (mother2 != -1) {
    mother2 += fMCIndexOffset;
  }
  if (daughter1 != -1) {
    daughter1 += fMCIndexOffset;
  }
  if (daughter2 != -1) {
    daughter2 += fMCIndexOffset;
  }

  /** if it is a K0/antiK0 to be tracked, convert it into K0s/K0L.
      
      NOTE: we could think of pushing the K0/antiK0 without tracking first
      and then push she K0s/K0L for tracking.
      In this way we would properly keep track of this conversion,
      but there is the risk of messing up with the indices, so this
      is not done for the time being.
  **/
  if (abs(pdgid) == 311 && doTracking) {
    LOG(WARN) << "K0/antiK0 requested for tracking: converting into K0s/K0L";
    pdgid = gRandom->Uniform() < 0.5 ? 310 : 130;
  }

  /** compute particle energy if negative **/
  if (e < 0) {
    double mass = particlePDG ? particlePDG->Mass() : 0.;
    e = std::sqrt(mass * mass + px * px + py * py + pz * pz);
  }

  /** add track to stack **/
  fStack->PushTrack(doTracking, mother1, pdgid, px, py, pz,
                    e, vx, vy, vz, tof, polx, poly, polz, proc, ntr,
                    weight, status, mother2);

  fNTracks++;
}

/*****************************************************************/

void PrimaryGenerator::setInteractionDiamond(const Double_t* xyz, const Double_t* sigmaxyz)
{
  /** set interaction diamond **/

  LOG(INFO) << "Setting interaction diamond: position = {"
            << xyz[0] << "," << xyz[1] << "," << xyz[2] << "} cm";
  LOG(INFO) << "Setting interaction diamond: width = {"
            << sigmaxyz[0] << "," << sigmaxyz[1] << "," << sigmaxyz[2] << "} cm";
  SetBeam(xyz[0], xyz[1], sigmaxyz[0], sigmaxyz[1]);
  SetTarget(xyz[2], sigmaxyz[2]);

  auto const& param = InteractionDiamondParam::Instance();
  SmearVertexXY(false);
  SmearVertexZ(false);
  SmearGausVertexXY(false);
  SmearGausVertexZ(false);
  if (param.distribution == o2::eventgen::EVertexDistribution::kFlat) {
    SmearVertexXY(true);
    SmearVertexZ(true);
  } else if (param.distribution == o2::eventgen::EVertexDistribution::kGaus) {
    SmearGausVertexXY(true);
    SmearGausVertexZ(true);
  } else {
    LOG(ERROR) << "PrimaryGenerator: Unsupported vertex distribution";
  }
}

/*****************************************************************/

void PrimaryGenerator::setInteractionVertex(const MCEventHeader* event)
{
  /** set interaction vertex **/

  Double_t xyz[3] = {event->GetX(), event->GetY(), event->GetZ()};
  SetBeam(xyz[0], xyz[1], 0., 0.);
  SetTarget(xyz[2], 0.);
  SmearVertexXY(false);
  SmearVertexZ(false);
  SmearGausVertexXY(false);
  SmearGausVertexZ(false);
}

/*****************************************************************/

Bool_t PrimaryGenerator::embedInto(TString fname)
{
  /** embed into **/

  /** check if a file is already open **/
  if (mEmbedFile && mEmbedFile->IsOpen()) {
    LOG(ERROR) << "Another embedding file is currently open";
    return kFALSE;
  }

  /** open file **/
  mEmbedFile = TFile::Open(fname);
  if (!mEmbedFile || !mEmbedFile->IsOpen()) {
    LOG(ERROR) << "Cannot open file for embedding: " << fname;
    return kFALSE;
  }

  /** get tree **/
  mEmbedTree = (TTree*)mEmbedFile->Get("o2sim");
  if (!mEmbedTree) {
    LOG(ERROR) << R"(Cannot find "o2sim" tree for embedding in )" << fname;
    return kFALSE;
  }

  /** get entries **/
  mEmbedEntries = mEmbedTree->GetEntries();
  if (mEmbedEntries <= 0) {
    LOG(ERROR) << "Invalid number of entries found in tree for embedding: " << mEmbedEntries;
    return kFALSE;
  }

  /** connect MC event header **/
  mEmbedEvent = new MCEventHeader;
  mEmbedTree->SetBranchAddress("MCEventHeader.", &mEmbedEvent);

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::PrimaryGenerator);
