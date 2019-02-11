/**************************************************************************
 * Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id$ */

//-------------------------------------------------------------------------
//     Class for Kinematic Events
//     Author: Andreas Morsch, CERN
//-------------------------------------------------------------------------
#include <TArrow.h>
#include <TMarker.h>
#include <TH2F.h>
#include <TTree.h>
#include <TFile.h>
#include <TParticle.h>
#include <TClonesArray.h>
#include <TList.h>
#include <TArrayF.h>

#include "AliLog.h"
#include "AliMCEvent.h"
#include "AliMCVertex.h"
#include "AliStack.h"
#include "AliTrackReference.h"
#include "AliHeader.h"
#include "AliGenEventHeader.h"
#include "AliGenHijingEventHeader.h"
#include "AliGenCocktailEventHeader.h"
#include "AliFastContainerAccess.h"
#include "AliMiscConstants.h"


Int_t AliMCEvent::fgkBgLabelOffset(10000000);


AliMCEvent::AliMCEvent():
    AliVEvent(),
    fStack(0),
    fMCParticles(0),
    fMCParticleMap(0),
    fHeader(new AliHeader()),
    fAODMCHeader(0),
    fTRBuffer(0),
    fTrackReferences(new TClonesArray("AliTrackReference", 1000)),
    fTreeTR(0),
    fTmpTreeTR(0),
    fTmpFileTR(0),
    fNprimaries(-1),
    fNparticles(-1),
    fSubsidiaryEvents(0),
    fPrimaryOffset(0),
    fSecondaryOffset(0),
    fExternal(0),
    fTopEvent(0),
    fVertex(0),
    fNBG(-1)
{
    // Default constructor
  fTopEvent = this;
}

AliMCEvent::AliMCEvent(const AliMCEvent& mcEvnt) :
    AliVEvent(mcEvnt),
    fStack(mcEvnt.fStack),
    fMCParticles(mcEvnt.fMCParticles),
    fMCParticleMap(mcEvnt.fMCParticleMap),
    fHeader(mcEvnt.fHeader),
    fAODMCHeader(mcEvnt.fAODMCHeader),
    fTRBuffer(mcEvnt.fTRBuffer),
    fTrackReferences(mcEvnt.fTrackReferences),
    fTreeTR(mcEvnt.fTreeTR),
    fTmpTreeTR(mcEvnt.fTmpTreeTR),
    fTmpFileTR(mcEvnt.fTmpFileTR),
    fNprimaries(mcEvnt.fNprimaries),
    fNparticles(mcEvnt.fNparticles),
    fSubsidiaryEvents(0),
    fPrimaryOffset(0),
    fSecondaryOffset(0),
    fExternal(0),
    fTopEvent(mcEvnt.fTopEvent),
    fVertex(mcEvnt.fVertex),
    fNBG(mcEvnt.fNBG)
{ 
// Copy constructor
}


AliMCEvent& AliMCEvent::operator=(const AliMCEvent& mcEvnt)
{
    // assignment operator
    if (this!=&mcEvnt) { 
	AliVEvent::operator=(mcEvnt); 
    }
  
    return *this; 
}

AliMCEvent::~AliMCEvent()
{
  if (fSubsidiaryEvents) delete fSubsidiaryEvents;
}

void AliMCEvent::ConnectTreeE (TTree* tree)
{
    // Connect the event header tree
    tree->SetBranchAddress("Header", &fHeader);
}

void AliMCEvent::ConnectTreeK (TTree* tree)
{
    // Connect Kinematics tree
    fStack = fHeader->Stack();
    fStack->ConnectTree(tree);
    //
    // Load the event
    fStack->GetEvent();
    
    UpdateEventInformation();
}

void AliMCEvent::ConnectHeaderAndStack(AliHeader* header)
{
  // fill MC event information from stack and header
  
  fHeader = header;
  fStack = fHeader->Stack();

  UpdateEventInformation();
}
 
void AliMCEvent::UpdateEventInformation()
{
    // bookkeeping for next event
  
    // Connect the kinematics tree to the stack
    if (!fMCParticles) fMCParticles = new TClonesArray("AliMCParticle",1000);

    // Initialize members
    fNparticles = fStack->GetNtrack();
    fNprimaries = fStack->GetNprimary();

    Int_t iev  = fHeader->GetEvent();
    Int_t ievr = fHeader->GetEventNrInRun();
    AliDebug(1, Form("AliMCEvent# %5d %5d: Number of particles: %5d (all) %5d (primaries)\n", 
		 iev, ievr, fNparticles, fNprimaries));
 
    // This is a cache for the TParticles converted to MCParticles on user request
    if (fMCParticleMap) {
	fMCParticleMap->Clear();
	fMCParticles->Delete();
	if (fNparticles>0) fMCParticleMap->Expand(fNparticles);
    }
    else
	fMCParticleMap = new TObjArray(fNparticles);
}

void AliMCEvent::ConnectTreeTR (TTree* tree)
{
    // Connect the track reference tree
    fTreeTR = tree;
    if (!fTreeTR) return; // just disconnect
    
    if (fTreeTR->GetBranch("AliRun")) {
	if (fTmpFileTR) {
	    fTmpFileTR->Close();
	    delete fTmpFileTR;
	}
	// This is an old format with one branch per detector not in synch with TreeK
	ReorderAndExpandTreeTR();
    } else {
	// New format 
	fTreeTR->SetBranchAddress("TrackReferences", &fTRBuffer);
    }
}

Int_t AliMCEvent::GetParticleAndTR(Int_t i, TParticle*& particle, TClonesArray*& trefs)
{
    // Retrieve entry i
  if (i >= BgLabelOffset()) {
    if (fSubsidiaryEvents) {
      AliMCEvent* bgEvent=0;
      i = FindIndexAndEvent(i, bgEvent);
      return bgEvent->GetParticleAndTR(i,particle,trefs);
    } else {
      particle = 0;
      trefs    = 0;
      return -1;
    }
  } 
  //  
  if (i < 0 || i >= fNparticles) {
    AliWarning(Form("AliMCEventHandler::GetEntry: Index out of range"));
	particle = 0;
	trefs    = 0;
	return (-1);
  }

  if (fSubsidiaryEvents) {
    AliMCEvent*   mc;
    Int_t idx = FindIndexAndEvent(i, mc);
    return mc->GetParticleAndTR(idx,particle,trefs);
  }
  //
  particle = fStack->Particle(i,kTRUE);
  if (fTreeTR) {
    fTreeTR->GetEntry(fStack->TreeKEntry(i,kTRUE));
    trefs    = fTRBuffer;
    return trefs->GetEntries();
  } else {
    trefs = 0;
    return -1;
  }

    
}


void AliMCEvent::Clean()
{
    // Clean-up before new trees are connected
    delete fStack; fStack = 0;

    // Clear TR
    if (fTRBuffer) {
	fTRBuffer->Delete();
	delete fTRBuffer;
	fTRBuffer = 0;
    }
}

#include <iostream>

void AliMCEvent::FinishEvent()
{
  // Clean-up after event
  //    
    if (fStack) fStack->Reset(0);
    fMCParticles->Delete();
    
    if (fMCParticleMap) 
      fMCParticleMap->Clear();
    if (fTRBuffer) {
      fTRBuffer->Delete();
    }
    //    fTrackReferences->Delete();
    fTrackReferences->Clear();
    fNparticles = -1;
    fNprimaries = -1;    
    fStack      =  0;
    delete fSubsidiaryEvents;
    fSubsidiaryEvents = 0;
    fNBG = -1;
}



void AliMCEvent::DrawCheck(Int_t i, Int_t search)
{
    //
    // Simple event display for debugging
    if (!fTreeTR) {
	AliWarning("No Track Reference information available");
	return;
    } 
    
    if (i > -1 && i < fNparticles) {
	fTreeTR->GetEntry(fStack->TreeKEntry(i));
    } else {
	AliWarning("AliMCEvent::GetEntry: Index out of range");
    }
    
    Int_t nh = fTRBuffer->GetEntries();
    
    
    if (search) {
	while(nh <= search && i < fNparticles - 1) {
	    i++;
	    fTreeTR->GetEntry(fStack->TreeKEntry(i));
	    nh =  fTRBuffer->GetEntries();
	}
	printf("Found Hits at %5d\n", i);
    }
    TParticle* particle = fStack->Particle(i,kTRUE);
    if (!particle) return;
    TH2F*    h = new TH2F("", "", 100, -500, 500, 100, -500, 500);
    Float_t x0 = particle->Vx();
    Float_t y0 = particle->Vy();

    Float_t x1 = particle->Vx() + particle->Px() * 50.;
    Float_t y1 = particle->Vy() + particle->Py() * 50.;
    
    TArrow*  a = new TArrow(x0, y0, x1, y1, 0.01);
    h->Draw();
    a->SetLineColor(2);
    
    a->Draw();
    
    for (Int_t ih = 0; ih < nh; ih++) {
	AliTrackReference* ref = (AliTrackReference*) fTRBuffer->At(ih);
	TMarker* m = new TMarker(ref->X(), ref->Y(), 20);
	m->Draw();
	m->SetMarkerSize(0.4);
	
    }
}


void AliMCEvent::ReorderAndExpandTreeTR()
{
//
//  Reorder and expand the track reference tree in order to match the kinematics tree.
//  Copy the information from different branches into one
//
//  TreeTR

    fTmpFileTR = new TFile("TrackRefsTmp.root", "recreate");
    fTmpTreeTR = new TTree("TreeTR", "TrackReferences");
    if (!fTRBuffer)  fTRBuffer = new TClonesArray("AliTrackReference", 100);
    fTmpTreeTR->Branch("TrackReferences", "TClonesArray", &fTRBuffer, 64000, 0);
    

//
//  Activate the used branches only. Otherwisw we get a bad memory leak.
    if (fTreeTR) {
	fTreeTR->SetBranchStatus("*",        0);
	fTreeTR->SetBranchStatus("AliRun.*", 1);
	fTreeTR->SetBranchStatus("ITS.*",    1);
	fTreeTR->SetBranchStatus("TPC.*",    1);
	fTreeTR->SetBranchStatus("TRD.*",    1);
	fTreeTR->SetBranchStatus("TOF.*",    1);
	fTreeTR->SetBranchStatus("FRAME.*",  1);
	fTreeTR->SetBranchStatus("MUON.*",   1);
    }
    
//
//  Connect the active branches
    TClonesArray* trefs[7];
    for (Int_t i = 0; i < 7; i++) trefs[i] = 0;
    if (fTreeTR){
	// make branch for central track references
	if (fTreeTR->GetBranch("AliRun")) fTreeTR->SetBranchAddress("AliRun", &trefs[0]);
	if (fTreeTR->GetBranch("ITS"))    fTreeTR->SetBranchAddress("ITS",    &trefs[1]);
	if (fTreeTR->GetBranch("TPC"))    fTreeTR->SetBranchAddress("TPC",    &trefs[2]);
	if (fTreeTR->GetBranch("TRD"))    fTreeTR->SetBranchAddress("TRD",    &trefs[3]);
	if (fTreeTR->GetBranch("TOF"))    fTreeTR->SetBranchAddress("TOF",    &trefs[4]);
	if (fTreeTR->GetBranch("FRAME"))  fTreeTR->SetBranchAddress("FRAME",  &trefs[5]);
	if (fTreeTR->GetBranch("MUON"))   fTreeTR->SetBranchAddress("MUON",   &trefs[6]);
    }

    Int_t np = fStack->GetNprimary();
    Int_t nt = fTreeTR->GetEntries();
    
    //
    // Loop over tracks and find the secondaries with the help of the kine tree
    Int_t ifills = 0;
    Int_t it     = 0;
    Int_t itlast = 0;
    TParticle* part;

    for (Int_t ip = np - 1; ip > -1; ip--) {
      part = fStack->Particle(ip,kTRUE);
//	printf("Particle %5d %5d %5d %5d %5d %5d \n", 
//	       ip, part->GetPdgCode(), part->GetFirstMother(), part->GetFirstDaughter(), 
//	       part->GetLastDaughter(), part->TestBit(kTransportBit));

	// Determine range of secondaries produced by this primary during transport	
	Int_t dau1  = part->GetFirstDaughter();
	if (dau1 < np) continue;  // This particle has no secondaries produced during transport
	Int_t dau2  = -1;
	if (dau1 > -1) {
	    Int_t inext = ip - 1;
	    while (dau2 < 0) {
		if (inext >= 0) {
     		    part = fStack->Particle(inext,kTRUE);
		    dau2 =  part->GetFirstDaughter();
		    if (dau2 == -1 || dau2 < np) {
			dau2 = -1;
		    } else {
			dau2--;
		    }
		} else {
		    dau2 = fStack->GetNtrack() - 1;
		}
		inext--;
	    } // find upper bound
	}  // dau2 < 0
	

//	printf("Check (1) %5d %5d %5d %5d %5d \n", ip, np, it, dau1, dau2);
//
// Loop over reference hits and find secondary label
// First the tricky part: find the entry in treeTR than contains the hits or
// make sure that no hits exist.
//
	Bool_t hasHits   = kFALSE;
	Bool_t isOutside = kFALSE;

	it = itlast;
	while (!hasHits && !isOutside && it < nt) {
	    fTreeTR->GetEntry(it++);
	    for (Int_t ib = 0; ib < 7; ib++) {
		if (!trefs[ib]) continue;
		Int_t nh = trefs[ib]->GetEntries();
		for (Int_t ih = 0; ih < nh; ih++) {
		    AliTrackReference* tr = (AliTrackReference*) trefs[ib]->At(ih);
		    Int_t label = tr->Label();
		    if (label >= dau1 && label <= dau2) {
			hasHits = kTRUE;
			itlast = it - 1;
			break;
		    }
		    if (label > dau2 || label < ip) {
			isOutside = kTRUE;
			itlast = it - 1;
			break;
		    }
		} // hits
		if (hasHits || isOutside) break;
	    } // branches
	} // entries

	if (!hasHits) {
	    // Write empty entries
	    for (Int_t id = dau1; (id <= dau2); id++) {
		fTmpTreeTR->Fill();
		ifills++;
	    } 
	} else {
	    // Collect all hits
	    fTreeTR->GetEntry(itlast);
	    for (Int_t id = dau1; (id <= dau2) && (dau1 > -1); id++) {
		for (Int_t ib = 0; ib < 7; ib++) {
		    if (!trefs[ib]) continue;
		    Int_t nh = trefs[ib]->GetEntries();
		    for (Int_t ih = 0; ih < nh; ih++) {
			AliTrackReference* tr = (AliTrackReference*) trefs[ib]->At(ih);
			Int_t label = tr->Label();
			// Skip primaries
			if (label == ip) continue;
			if (label > dau2 || label < dau1) 
			    printf("AliMCEventHandler::Track Reference Label out of range !: %5d %5d %5d %5d \n", 
				   itlast, label, dau1, dau2);
			if (label == id) {
			    // secondary found
			    tr->SetDetectorId(ib-1);
			    Int_t nref =  fTRBuffer->GetEntriesFast();
			    TClonesArray &lref = *fTRBuffer;
			    new(lref[nref]) AliTrackReference(*tr);
			}
		    } // hits
		} // branches
		fTmpTreeTR->Fill();
		fTRBuffer->Delete();
		ifills++;
	    } // daughters
	} // has hits
    } // tracks

    //
    // Now loop again and write the primaries
    //
    it = nt - 1;
    for (Int_t ip = 0; ip < np; ip++) {
	Int_t labmax = -1;
	while (labmax < ip && it > -1) {
	    fTreeTR->GetEntry(it--);
	    for (Int_t ib = 0; ib < 7; ib++) {
		if (!trefs[ib]) continue;
		Int_t nh = trefs[ib]->GetEntries();
		// 
		// Loop over reference hits and find primary labels
		for (Int_t ih = 0; ih < nh; ih++) {
		    AliTrackReference* tr = (AliTrackReference*)  trefs[ib]->At(ih);
		    Int_t label = tr->Label();
		    if (label < np && label > labmax) {
			labmax = label;
		    }
		    
		    if (label == ip) {
			tr->SetDetectorId(ib-1);
			Int_t nref = fTRBuffer->GetEntriesFast();
			TClonesArray &lref = *fTRBuffer;
			new(lref[nref]) AliTrackReference(*tr);
		    }
		} // hits
	    } // branches
	} // entries
	it++;
	fTmpTreeTR->Fill();
	fTRBuffer->Delete();
	ifills++;
    } // tracks
    // Check


    // Clean-up
    delete fTreeTR; fTreeTR = 0;
    
    for (Int_t ib = 0; ib < 7; ib++) {
	if (trefs[ib]) {
	    trefs[ib]->Clear();
	    delete trefs[ib];
	    trefs[ib] = 0;
	}
    }

    if (ifills != fStack->GetNtrack()) 
	printf("AliMCEvent:Number of entries in TreeTR (%5d) unequal to TreeK (%5d) \n", 
	       ifills, fStack->GetNtrack());

    fTmpTreeTR->Write();
    fTreeTR = fTmpTreeTR;
}

Bool_t AliMCEvent::IsFromSubsidiaryEvent(int id) const
{
  // returns true if particle id is from subsidiary (to which the signal was embedded) event
  if (id >= BgLabelOffset() && fSubsidiaryEvents) return kTRUE;
  if (fSubsidiaryEvents) {
    AliMCEvent* mc;
    FindIndexAndEvent(id, mc);
    if (mc != fSubsidiaryEvents->At(0)) return kTRUE;
  } 
  return kFALSE;
}


AliVParticle* AliMCEvent::GetTrack(Int_t i) const
{
    // Get MC Particle i
    //

    if (fExternal) {
	return ((AliVParticle*) (fMCParticles->At(i)));
    }
    
    //
    // Check first if this explicitely accesses the subsidiary event
    
    if (i >= BgLabelOffset()) {
	if (fSubsidiaryEvents) {
	  AliMCEvent* bgEvent=0;
	  i = FindIndexAndEvent(i, bgEvent);
	  return bgEvent->GetTrack(i);
	} else {
	  return (AliVParticle*)GetDummyTrack();
	}
    }
    
    //
    AliMCParticle *mcParticle = 0;
    TParticle     *particle   = 0;
    TClonesArray  *trefs      = 0;
    Int_t          ntref      = 0;
    TObjArray     *rarray     = 0;

    // Out of range check
    if (i < 0 || i >= fNparticles) {
      if (i==gkDummyLabel) return 0;
      AliWarning(Form("AliMCEvent::GetEntry: Index out of range"));
      mcParticle = 0;
      return (mcParticle);
    }

    
    if (fSubsidiaryEvents) {
	AliMCEvent*   mc;
	Int_t idx = FindIndexAndEvent(i, mc);
	return (mc->GetTrack(idx));
    } 

    //
    // First check If the MC Particle has been already cached
    if(!fMCParticleMap->At(i)) {
      // Get particle from the stack
      particle   = fStack->Particle(i,kTRUE);
      // Get track references from Tree TR
      if (fTreeTR) {
	fTreeTR->GetEntry(fStack->TreeKEntry(i,kTRUE));
	trefs     = fTRBuffer;
	ntref     = trefs->GetEntriesFast();
	rarray    = new TObjArray(ntref);
	Int_t nen = fTrackReferences->GetEntriesFast();
	for (Int_t j = 0; j < ntref; j++) {
	  // Save the track references in a TClonesArray
	  AliTrackReference* ref = dynamic_cast<AliTrackReference*>((*fTRBuffer)[j]);
	  // Save the pointer in a TRefArray
	  if (ref) {
	    new ((*fTrackReferences)[nen]) AliTrackReference(*ref);
	    rarray->AddAt((*fTrackReferences)[nen], j);
	    nen++;
	  }
	} // loop over track references for entry i
      } // if TreeTR available
      Int_t nentries = fMCParticles->GetEntriesFast();
      mcParticle = new ((*fMCParticles)[nentries]) AliMCParticle(particle, rarray, i);
      fMCParticleMap->AddAt(mcParticle, i);
      if (mcParticle) {
	TParticle* part = mcParticle->Particle();
	Int_t imo  = part->GetFirstMother();
	Int_t id1  = part->GetFirstDaughter();
	Int_t id2  = part->GetLastDaughter();
       mcParticle->SetLabel(i); // RS mcParticle should refer to its position in its parent stack
       mcParticle->SetStack(fStack);
	if (fPrimaryOffset > 0 || fSecondaryOffset > 0) {
	  // Remapping of the mother and daughter indices
	  if (imo>=0) {
	    mcParticle->SetMother( imo<fNprimaries ? imo + fPrimaryOffset : imo + fSecondaryOffset - fNprimaries);
	  }
	  if (id1>=0) {		
	    if (id1 < fNprimaries) {
	      mcParticle->SetFirstDaughter(id1 + fPrimaryOffset);
	      mcParticle->SetLastDaughter (id2 + fPrimaryOffset);
	    } else {
	      mcParticle->SetFirstDaughter(id1 + fSecondaryOffset - fNprimaries);
	      mcParticle->SetLastDaughter (id2 + fSecondaryOffset - fNprimaries);
	    }
	  }
	  //
	  /* // RS: this breacks convention on label
	  if (i > fNprimaries) {
	    mcParticle->SetLabel(i + fPrimaryOffset);
	  } else {
	    mcParticle->SetLabel(i + fSecondaryOffset - fNprimaries);
	  }
	  */
	} else {
	  mcParticle->SetFirstDaughter(id1);
	  mcParticle->SetLastDaughter (id2);
	  mcParticle->SetMother       (imo);
	}
      }
    } else {
      mcParticle = dynamic_cast<AliMCParticle*>(fMCParticleMap->At(i));
    }
    
    //Printf("mcParticleGetMother %d",mcParticle->GetMother());
    return mcParticle;
}

TParticle* AliMCEvent::ParticleFromStack(Int_t i) const
{
  // Get MC Particle i from original stack, accounting for eventual label offset in case of embedding
  //
  if (fExternal) {
    AliMCParticle* mcp = (AliMCParticle*)fMCParticles->At(i);
    return mcp ? mcp->Particle() : 0;
  }
  if (fSubsidiaryEvents) {
    AliMCEvent* event = (AliMCEvent*)fSubsidiaryEvents->At(i/BgLabelOffset());
    return event->Stack()->Particle(i%BgLabelOffset(),kTRUE);
  }
  return fStack->Particle(i,kTRUE);
}

AliGenEventHeader* AliMCEvent::GenEventHeader() const 
{
  if (!fExternal) {
    // ESD
    return (fHeader->GenEventHeader());
  } else {
    // AOD
    if (fAODMCHeader) {
      TList * lh = fAODMCHeader->GetCocktailHeaders();
      if (lh) {return ((AliGenEventHeader*) lh->At(0));}
    }
  }
  return 0;
}


void AliMCEvent::AddSubsidiaryEvent(AliMCEvent* event) 
{
    // Add a subsidiary event to the list; for example merged background event.
    if (!fSubsidiaryEvents) {
      TList* events = new TList();
      events->SetOwner(kFALSE);
      events->Add(new AliMCEvent(*this));
      fSubsidiaryEvents = events;
    }
    
    fSubsidiaryEvents->Add(event);
    if (fStack) fStack->SetMCEmbeddingFlag(kTRUE);
    event->SetTopEvent(this);
}

AliGenEventHeader *AliMCEvent::FindHeader(Int_t ipart) {
  //
  // Get Header belonging to this track; 
  // only works for primaries (i.e. particles coming from the Generator)
  // Also sorts out the case of Cocktail event (get header of subevent in cocktail generetor header)  
  //

  AliMCEvent *event = this;

  if (fSubsidiaryEvents) {
    // Get pointer to subevent if needed
    ipart = FindIndexAndEvent(ipart,event); 
  }

  AliGenEventHeader* header = event->GenEventHeader();
  if (ipart >= header->NProduced()) {
    AliWarning(Form("Not a primary -- returning 0 (idx %d, nPrimary %d)",ipart,header->NProduced()));
    return 0;
  }
  AliGenCocktailEventHeader *coHeader = dynamic_cast<AliGenCocktailEventHeader*>(header);
  if (coHeader) { // Cocktail event
    TList* headerList = coHeader->GetHeaders();
    TIter headIt(headerList);
    Int_t nproduced = 0;
    do { // Go trhough all headers and look for the correct one
      header = (AliGenEventHeader*) headIt();
      if (header) nproduced += header->NProduced();
    } while (header && ipart >= nproduced);
  }

  return header;
}

Int_t AliMCEvent::FindIndexAndEvent(Int_t oldidx, AliMCEvent*& event) const
{
    // Find the index and event in case of composed events like signal + background
  
  // Check first if this explicitely accesses the subsidiary event
  if (oldidx >= BgLabelOffset()) {
    event = (AliMCEvent*) (fSubsidiaryEvents->At(oldidx/BgLabelOffset()));
    return oldidx%BgLabelOffset();
  }
  if (fSubsidiaryEvents) {
    TIter next(fSubsidiaryEvents);
    next.Reset();
    if (oldidx < fNprimaries) {
      while((event = (AliMCEvent*)next())) {
	if (oldidx < (event->GetPrimaryOffset() + event->GetNumberOfPrimaries())) break;
      }
      if (event) {
	return (oldidx - event->GetPrimaryOffset());
      } else {
	return (-1);
      }
    } else {
      while((event = (AliMCEvent*)next())) {
	if (oldidx < (event->GetSecondaryOffset() + (event->GetNumberOfTracks() - event->GetNumberOfPrimaries()))) break;
      }
      if (event) {
	return (oldidx - event->GetSecondaryOffset() + event->GetNumberOfPrimaries());
      } else {
	return (-1);
      }
    }
  } else {
    return oldidx;
  }
}

Int_t AliMCEvent::BgLabelToIndex(Int_t label)
{
    // Convert a background label to an absolute index
    if (fSubsidiaryEvents) {
	AliMCEvent* bgEvent = (AliMCEvent*) (fSubsidiaryEvents->At(1));
	label -= BgLabelOffset();
	if (label < bgEvent->GetNumberOfPrimaries()) {
	    label += bgEvent->GetPrimaryOffset();
	} else {
	    label += (bgEvent->GetSecondaryOffset() - fNprimaries);
	}
    }
    return (label);
}


Bool_t AliMCEvent::IsPhysicalPrimary(Int_t i) const
{
//
// Delegate to subevent if necesarry 

    
    if (!fSubsidiaryEvents) {      
      return (i >= BgLabelOffset()) ? kFALSE : fStack->IsPhysicalPrimary(i,kTRUE);
    } else {
	AliMCEvent* evt = 0;
	Int_t idx = FindIndexAndEvent(i, evt);
	return (evt->IsPhysicalPrimary(idx));
    }
}

Bool_t AliMCEvent::IsSecondaryFromWeakDecay(Int_t i)
{
//
// Delegate to subevent if necesarry 
    if (!fSubsidiaryEvents) {
      return (i >= BgLabelOffset()) ? kFALSE : fStack->IsSecondaryFromWeakDecay(i,kTRUE);
    } else {
	AliMCEvent* evt = 0;
	Int_t idx = FindIndexAndEvent(i, evt);
	return (evt->IsSecondaryFromWeakDecay(idx));
    }
}

Bool_t AliMCEvent::IsSecondaryFromMaterial(Int_t i)
{
//
// Delegate to subevent if necesarry 
    if (!fSubsidiaryEvents) {
      return (i >= BgLabelOffset()) ? kFALSE : fStack->IsSecondaryFromMaterial(i,kTRUE);
    } else {
	AliMCEvent* evt = 0;
	Int_t idx = FindIndexAndEvent(i, evt);
	return (evt->IsSecondaryFromMaterial(idx));
    }
}


void AliMCEvent::InitEvent()
{
//
// Initialize the subsidiary event structure
    if (fSubsidiaryEvents) {
	TIter next(fSubsidiaryEvents);
	AliMCEvent* evt;
	fNprimaries = 0;
	fNparticles = 0;
	
	while((evt = (AliMCEvent*)next())) {
	    fNprimaries += evt->GetNumberOfPrimaries();	
	    fNparticles += evt->GetNumberOfTracks();    
	}
	
	Int_t ioffp = 0;
	Int_t ioffs = fNprimaries;
	next.Reset();
	
	while((evt = (AliMCEvent*)next())) {
	    evt->SetPrimaryOffset(ioffp);
	    evt->SetSecondaryOffset(ioffs);
	    ioffp += evt->GetNumberOfPrimaries();
	    ioffs += (evt->GetNumberOfTracks() - evt->GetNumberOfPrimaries());	    
	}
    }
}

void AliMCEvent::PreReadAll()                              
{
    // Preread the MC information
  if (fSubsidiaryEvents) { // prereading should be done only once all sub events read and initialized
    TIter next(fSubsidiaryEvents);
    AliMCEvent* evt;
    while((evt = (AliMCEvent*)next())) {
      evt->PreReadAll();
    }
    return;
  }
  
  
    Int_t i;
    // secondaries
    for (i = fStack->GetNprimary(); i < fStack->GetNtrack(); i++) 
    {
	GetTrack(i);
    }
    // primaries
    for (i = 0; i < fStack->GetNprimary(); i++) 
    {
	GetTrack(i);
    }
    AssignGeneratorIndex();
}

const AliVVertex * AliMCEvent::GetPrimaryVertex() const 
{
    // Create a MCVertex object from the MCHeader information
    TArrayF v;
    GenEventHeader()->PrimaryVertex(v) ;
    if (!fVertex) {
	fVertex = new AliMCVertex(v[0], v[1], v[2]);
    } else {
	((AliMCVertex*) fVertex)->SetPosition(v[0], v[1], v[2]);
    }
    return fVertex;
}

Bool_t AliMCEvent::IsFromBGEvent(Int_t index)
{
    // Checks if a particle is from the background events
    // Works for HIJING inside Cocktail
  if (index >= BgLabelOffset() && !fSubsidiaryEvents) return kTRUE;
    if (fNBG == -1) {
	AliGenCocktailEventHeader* coHeader = 
	    dynamic_cast<AliGenCocktailEventHeader*> (GenEventHeader());
	if (!coHeader) return (0);
	TList* list = coHeader->GetHeaders();
	AliGenHijingEventHeader* hijingH = dynamic_cast<AliGenHijingEventHeader*>(list->FindObject("Hijing"));
	if (!hijingH) return (0);
	fNBG = hijingH->NProduced();
    }
    
    return (index < fNBG);
}


TList* AliMCEvent::GetCocktailList()
{
  //gives the CocktailHeaders when reading ESDs/AODs (corresponding to fExteral=kFALSE/kTRUE)
  //the AODMC header (and the aodmc array) is passed as an instance to MCEvent by the AliAODInputHandler
  if(fExternal==kFALSE) { 
    AliGenCocktailEventHeader* coHeader =dynamic_cast<AliGenCocktailEventHeader*> (GenEventHeader());
    if(!coHeader) {
      return 0;
    } else {
      return (coHeader->GetHeaders());
    }
  } else {
    if(!fAODMCHeader) { 
      return 0;
    } else {
      return (fAODMCHeader->GetCocktailHeaders());
    }
  }
}


TString AliMCEvent::GetGenerator(Int_t index)
{
  if (index >= BgLabelOffset() && !fSubsidiaryEvents) {
    TString retv = "suppressed";
    return retv;
  }
  Int_t nsumpart=fNprimaries;
  TList* lh = GetCocktailList();
  if(!lh){ TString noheader="nococktailheader";
    return noheader;}
  Int_t nh=lh->GetEntries();
  for (Int_t i = nh-1; i >= 0; i--){
    AliGenEventHeader* gh=(AliGenEventHeader*)lh->At(i);
    TString genname=gh->GetName();
    Int_t npart=gh->NProduced();
    if (i == 0) npart = nsumpart;
    if(index < nsumpart && index >= (nsumpart-npart)) return genname;
    nsumpart-=npart;
  }
  TString empty="";
  return empty;
}

void AliMCEvent::AssignGeneratorIndex() {
  //
  // Assign the generator index to each particle
  //
  TList* list = GetCocktailList();
  if (fNprimaries <= 0) {
    AliWarning(Form("AliMCEvent::AssignGeneratorIndex: no primaries %10d\n", fNprimaries));
    return;
}
  if (!list) {
    return;
  } else {
    Int_t nh = list->GetEntries();
    Int_t nsumpart = fNprimaries;
    for(Int_t i = nh-1; i >= 0; i--){
      AliGenEventHeader* gh = (AliGenEventHeader*)list->At(i);
      Int_t npart = gh->NProduced();
      if (i==0) {
	if (npart != nsumpart) {
	  //	  printf("Header inconsistent ! %5d %5d \n", npart, nsumpart);
	}
	npart = nsumpart;
      }
      //
      // Loop over primary particles for generator i
      for (Int_t j = nsumpart-1; j >= nsumpart-npart; j--) {
	AliVParticle* part = fTopEvent->GetTrack(j); // after 1st GetTrack indices correspond to top event
	if (!part) {
	  AliWarning(Form("AliMCEvent::AssignGeneratorIndex: 0-pointer to particle j %8d npart %8d nsumpart %8d Nprimaries %8d\n", 
			  j, npart, nsumpart, fNprimaries));
	  break;
	}
	part->SetGeneratorIndex(i);
	Int_t dmin = part->GetFirstDaughter();
	Int_t dmax = part->GetLastDaughter();
	if (dmin == -1) continue;
	AssignGeneratorIndex(i, dmin, dmax);
      } 
      nsumpart -= npart;
    }
  }
}
void AliMCEvent::AssignGeneratorIndex(Int_t index, Int_t dmin, Int_t dmax) {
  for (Int_t k = dmin; k <= dmax; k++) {
    AliVParticle* dpart = fTopEvent->GetTrack(k);
    dpart->SetGeneratorIndex(index);
    Int_t d1 = dpart->GetFirstDaughter();
    Int_t d2 = dpart->GetLastDaughter();
    if (d1 > -1) {
      AssignGeneratorIndex(index, d1, d2);
    }
  }
}

Bool_t  AliMCEvent::GetCocktailGenerator(Int_t index,TString &nameGen){
     //method that gives the generator for a given particle with label index (or that of the corresponding primary)
     AliVParticle* mcpart0 = (AliVParticle*) (fTopEvent->GetTrack(index));
     if(!mcpart0){
       printf("AliMCEvent-BREAK: No valid AliMCParticle at label %i\n",index);
       return 0;
     }
     /*
     Int_t ig = mcpart0->GetGeneratorIndex();
     if (ig != -1) {
       nameGen = ((AliGenEventHeader*)GetCocktailList()->At(ig))->GetName();
       return 1;
     }
     */
    nameGen=GetGenerator(index);
    if(nameGen.Contains("nococktailheader") )return 0;
    Int_t lab=index;

    while(nameGen.IsWhitespace()){
      
      
    AliVParticle* mcpart = (AliVParticle*) (fTopEvent->GetTrack(lab));
 
     if(!mcpart){
      printf("AliMCEvent-BREAK: No valid AliMCParticle at label %i\n",lab);
      break;}
     Int_t mother=0;
     mother = mcpart->GetMother();
   
    if(mother<0){
      printf("AliMCEvent - BREAK: Reached primary particle without valid mother\n");
      break;
    }
      AliVParticle* mcmom = (AliVParticle*) (fTopEvent->GetTrack(mother));
      if(!mcmom){
      printf("AliMCEvent-BREAK: No valid AliMCParticle mother at label %i\n",mother);
       break;
      }
      lab=mother;
   
    nameGen=GetGenerator(mother);
   }
   
   return 1;
}

void  AliMCEvent::SetParticleArray(TClonesArray* mcParticles) 
  {
    fMCParticles = mcParticles; 
    fNparticles = fMCParticles->GetEntries(); 
    fExternal = kTRUE; 
    fNprimaries = 0;
    struct Local {
      static Int_t binaryfirst(TClonesArray* a, Int_t low, Int_t high)
      {
	Int_t mid  = low + (high - low)/2;
	if (low > a->GetEntries()-1) return (a->GetEntries()-1);
	if (!((AliVParticle*) a->At(mid))->IsPrimary()) {
	  if (mid > 1 && !((AliVParticle*) a->At(mid-1))->IsPrimary()) {
	    return binaryfirst(a, low, mid-1);
	  } else {
	    return mid;
	  } 
	} else {
	  return binaryfirst(a, mid+1, high);
	}
      }
    };
    fNprimaries = Local::binaryfirst(mcParticles, 0, mcParticles->GetEntries()-1);
    AssignGeneratorIndex();
  }

AliVEvent::EDataLayoutType AliMCEvent::GetDataLayoutType() const
{
  return AliVEvent::kMC;
}

TParticle* AliMCEvent::Particle(int i) const
{
  // extract Particle from the MCTrack with global index i
  const AliMCParticle* mcpart = (const AliMCParticle*)GetTrack(i);
  return mcpart ? mcpart->Particle() : 0;
}

Int_t AliMCEvent::Raw2MergedLabel(int lbRaw) const
{
  // convert raw label corresponding to stack and eventual embedded MC component to global
  // label corresponding to MCEvent::GetTrack conventions (first all primiraies then all secondaries)
  if (!fSubsidiaryEvents) return lbRaw;
  int lb = lbRaw%BgLabelOffset();
  AliMCEvent* mcev = (AliMCEvent*)fSubsidiaryEvents->At(lbRaw/BgLabelOffset());
  int nprim = mcev->GetNumberOfPrimaries();
  lb += lb<nprim ? mcev->GetPrimaryOffset() : mcev->GetSecondaryOffset() - nprim;
  return lb;
}

//_____________________________________________________________________________
Int_t AliMCEvent::GetPrimary(Int_t id)
{
  //
  // Return number of primary that has generated track
  //
  
  int current, parent;
  //
  parent=id;
  while (1) {
    current=parent;
    parent=Particle(current)->GetFirstMother();
    if(parent<0) return current;
  }
}

//_________________________________________________________________
AliMCParticle* AliMCEvent::GetDummyTrack()
{
  static AliMCParticle dummy(AliStack::GetDummyParticle(), 0, gkDummyLabel);
  return &dummy;
}

ClassImp(AliMCEvent)
