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
                                Int_t parent, Bool_t wanttracking,
                                Double_t e, Double_t tof,
                                Double_t weight, TMCProcess proc)
{
  /** add track **/

  /** check if particle exists in PDG database **/
  if (!TDatabasePDG::Instance()->GetParticle(pdgid)) {
    LOG(WARN) << "Skipping particle undefined in PDG: pdg = " << pdgid;
    return;
  }

  /** success **/
  FairPrimaryGenerator::AddTrack(pdgid, px, py, pz, vx, vy, vz, parent, wanttracking, e, tof, weight, proc);
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
  SmearVertexXY(false);
  SmearVertexZ(false);
  SmearGausVertexXY(true);
  SmearGausVertexZ(true);
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
