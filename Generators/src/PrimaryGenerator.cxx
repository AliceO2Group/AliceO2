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
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/GeneratorHeader.h"
#include "FairGenericStack.h"
#include "FairGenerator.h"
#include "FairMCEventHeader.h"
#include "FairLogger.h"
#include "TFile.h"
#include "TTree.h"

using o2::dataformats::MCEventHeader;
using o2::dataformats::GeneratorHeader;

namespace o2
{
namespace eventgen
{

/*****************************************************************/
/*****************************************************************/

PrimaryGenerator::PrimaryGenerator() : FairPrimaryGenerator("ALICEo2", "ALICEo2 Primary Generator")
{
  /** default constructor **/
}

/*****************************************************************/

PrimaryGenerator::~PrimaryGenerator() { /** default destructor **/}

/*****************************************************************/

Bool_t PrimaryGenerator::GenerateEvent(FairGenericStack* pStack)
{
  /** generate event **/

  /** normal generation if no embedding **/
  if (!mEmbedTree)
    return FairPrimaryGenerator::GenerateEvent(pStack);

  /** this is for embedding **/

  /** setup interaction diamond **/
  mEmbedTree->GetEntry(mEmbedCounter);
  setInteractionDiamond(mEmbedEvent);

  /** generate event **/
  if (!FairPrimaryGenerator::GenerateEvent(pStack))
    return kFALSE;

  /** add embedding info to event header **/
  auto o2event = dynamic_cast<MCEventHeader*>(fEvent);
  if (o2event) {
    o2event->setEmbeddingFileName(mEmbedFile->GetName());
    o2event->setEmbeddingEventCounter(mEmbedCounter);
  }

  /** increment embedding counter **/
  mEmbedCounter++;
  mEmbedCounter %= mEmbedEntries;

  /** success **/
  return kTRUE;
}

/*****************************************************************/

void PrimaryGenerator::addHeader(GeneratorHeader* header)
{
  /** add header **/

  /** setup header **/
  header->setTrackOffset(fMCIndexOffset);
  header->setNumberOfTracks(fNTracks - fMCIndexOffset);

  /** check o2 event header **/
  auto o2event = dynamic_cast<MCEventHeader*>(fEvent);
  if (!o2event)
    return;
  o2event->addHeader(header);
}

/*****************************************************************/

void PrimaryGenerator::setInteractionDiamond(const Double_t* xyz, const Double_t* sigmaxyz, Bool_t smear)
{
  /** set interaction diamond **/

  SetBeam(xyz[0], xyz[1], sigmaxyz[0], sigmaxyz[1]);
  SetTarget(xyz[2], sigmaxyz[2]);
  SmearVertexXY(kFALSE);
  SmearVertexZ(kFALSE);
  SmearGausVertexXY(smear);
  SmearGausVertexZ(smear);
}

/*****************************************************************/

void PrimaryGenerator::setInteractionDiamond(const FairMCEventHeader* event)
{
  /** set interaction diamond **/

  Double_t xyz[3] = { event->GetX(), event->GetY(), event->GetZ() };
  Double_t sigmaxyz[3] = { 0., 0., 0. };
  setInteractionDiamond(xyz, sigmaxyz, kFALSE);
}

/*****************************************************************/

Bool_t PrimaryGenerator::embedInto(TString fname)
{
  /** embed into **/

  /** check if a file is already open **/
  if (mEmbedFile && mEmbedFile->IsOpen()) {
    LOG(ERROR) << "Another embedding file is currently open" << std::endl;
    return kFALSE;
  }

  /** open file **/
  mEmbedFile = TFile::Open(fname);
  if (!mEmbedFile || !mEmbedFile->IsOpen()) {
    LOG(ERROR) << "Cannot open file for embedding: " << fname << std::endl;
    return kFALSE;
  }

  /** get tree **/
  mEmbedTree = (TTree*)mEmbedFile->Get("o2sim");
  if (!mEmbedTree) {
    LOG(ERROR) << R"(Cannot find "o2sim" tree for embedding in )" << fname << std::endl;
    return kFALSE;
  }

  /** get entries **/
  mEmbedEntries = mEmbedTree->GetEntries();
  if (mEmbedEntries <= 0) {
    LOG(ERROR) << "Invalid number of entries found in tree for embedding: " << mEmbedEntries << std::endl;
    return kFALSE;
  }

  /** connect MC event header **/
  TBranch* theBranch = mEmbedTree->GetBranch("MCEventHeader.");
  TClass* theClass = new TClass();
  EDataType theType;
  theBranch->GetExpectedType(theClass, theType);
  mEmbedEvent = (FairMCEventHeader*)theClass->New();
  mEmbedTree->SetBranchAddress("MCEventHeader.", &mEmbedEvent);

  /** success **/
  return kTRUE;
}

/*****************************************************************/
/*****************************************************************/

} /* namespace eventgen */
} /* namespace o2 */

ClassImp(o2::eventgen::PrimaryGenerator)
