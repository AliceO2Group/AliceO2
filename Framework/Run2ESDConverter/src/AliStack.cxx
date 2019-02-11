/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  Particles stack class                                                    //
//  Implements the TMCVirtualStack of the Virtual Monte Carlo                //
//  Holds the particles transported during simulation                        //
//  Is used to compare results of reconstruction with simulation             //
//  Author A.Morsch                                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

 
#include <TClonesArray.h>
#include <TObjArray.h>
#include <TPDGCode.h>
#include <TMCProcess.h>
#include <TParticle.h>
#include <TParticlePDG.h>
#include <TDatabasePDG.h>
#include <TTree.h>
#include <TDirectory.h>

#include "AliLog.h"
#include "AliStack.h"
#include "AliMiscConstants.h"


ClassImp(AliStack)


const Char_t* AliStack::fgkEmbedPathsKey = "embeddingBKGPaths";

//_______________________________________________________________________
AliStack::AliStack():
  fParticles("TParticle", 1000),
  fParticleMap(),
  fParticleFileMap(0),
  fParticleBuffer(0),
  fCurrentTrack(0),
  fTreeK(0),
  fNtrack(0),
  fNprimary(0),
  fCurrent(-1),
  fCurrentPrimary(-1),
  fHgwmk(0),
  fLoadPoint(0),
  fTrackLabelMap(0),
  fMCEmbeddingFlag(kFALSE)
{
  //
  // Default constructor
  //
}

//_______________________________________________________________________
AliStack::AliStack(Int_t size, const char* /*evfoldname*/):
  fParticles("TParticle",1000),
  fParticleMap(size),
  fParticleFileMap(0),
  fParticleBuffer(0),
  fCurrentTrack(0),
  fTreeK(0),
  fNtrack(0),
  fNprimary(0),
  fNtransported(0),
  fCurrent(-1),
  fCurrentPrimary(-1),
  fHgwmk(0),
  fLoadPoint(0),
  fTrackLabelMap(0),
  fMCEmbeddingFlag(kFALSE)
{
  //
  //  Constructor
  //
}

//_______________________________________________________________________
AliStack::AliStack(const AliStack& st):
    TVirtualMCStack(st),
    fParticles("TParticle",1000),
    fParticleMap(*(st.Particles())),
    fParticleFileMap(st.fParticleFileMap),
    fParticleBuffer(0),
    fCurrentTrack(0),
    fTreeK((TTree*)(st.fTreeK->Clone())),
    fNtrack(st.GetNtrack()),
    fNprimary(st.GetNprimary()),
    fNtransported(st.GetNtransported()),
    fCurrent(-1),
    fCurrentPrimary(-1),
    fHgwmk(0),
    fLoadPoint(0),
    fTrackLabelMap(0),
    fMCEmbeddingFlag(kFALSE)
{
    // Copy constructor
}


//_______________________________________________________________________
void AliStack::Copy(TObject&) const
{
  AliFatal("Not implemented!");
}

//_______________________________________________________________________
AliStack::~AliStack()
{
  //
  // Destructor
  //
  
    fParticles.Clear();
}

//
// public methods
//

//_____________________________________________________________________________
void AliStack::PushTrack(Int_t done, Int_t parent, Int_t pdg, const Float_t *pmom,
                        const Float_t *vpos, const Float_t *polar, Float_t tof,
                        TMCProcess mech, Int_t &ntr, Float_t weight, Int_t is)
{ 
  //
  // Load a track on the stack
  //
  // done     1 if the track has to be transported
  //          0 if not
  // parent   identifier of the parent track. -1 for a primary
  // pdg    particle code
  // pmom     momentum GeV/c
  // vpos     position 
  // polar    polarisation 
  // tof      time of flight in seconds
  // mecha    production mechanism
  // ntr      on output the number of the track stored
  //

  //  const Float_t tlife=0;
  
  //
  // Here we get the static mass
  // For MC is ok, but a more sophisticated method could be necessary
  // if the calculated mass is required
  // also, this method is potentially dangerous if the mass
  // used in the MC is not the same of the PDG database
  //
    TParticlePDG* pmc =  TDatabasePDG::Instance()->GetParticle(pdg);
    if (pmc) {
	Float_t mass = TDatabasePDG::Instance()->GetParticle(pdg)->Mass();
	Float_t e=TMath::Sqrt(mass*mass+pmom[0]*pmom[0]+
			      pmom[1]*pmom[1]+pmom[2]*pmom[2]);
	
//    printf("Loading  mass %f ene %f No %d ip %d parent %d done %d pos %f %f %f mom %f %f %f kS %d m \n",
//	   mass,e,fNtrack,pdg,parent,done,vpos[0],vpos[1],vpos[2],pmom[0],pmom[1],pmom[2],kS);
  

	PushTrack(done, parent, pdg, pmom[0], pmom[1], pmom[2], e,
		 vpos[0], vpos[1], vpos[2], tof, polar[0], polar[1], polar[2],
		 mech, ntr, weight, is);
    } else {
	AliWarning(Form("Particle type %d not defined in PDG Database !", pdg));
	AliWarning("Particle skipped !");
    }
}

//_____________________________________________________________________________
void AliStack::PushTrack(Int_t done, Int_t parent, Int_t pdg,
  	              Double_t px, Double_t py, Double_t pz, Double_t e,
  		      Double_t vx, Double_t vy, Double_t vz, Double_t tof,
		      Double_t polx, Double_t poly, Double_t polz,
		      TMCProcess mech, Int_t &ntr, Double_t weight, Int_t is)
{ 
  //
  // Load a track on the stack
  //
  // done        1 if the track has to be transported
  //             0 if not
  // parent      identifier of the parent track. -1 for a primary
  // pdg         particle code
  // kS          generation status code
  // px, py, pz  momentum GeV/c
  // vx, vy, vz  position 
  // polar       polarisation 
  // tof         time of flight in seconds
  // mech        production mechanism
  // ntr         on output the number of the track stored
  //    
  // New method interface: 
  // arguments were changed to be in correspondence with TParticle
  // constructor.
  // Note: the energy is not calculated from the static mass but
  // it is passed by argument e.

  const Int_t kFirstDaughter=-1;
  const Int_t kLastDaughter=-1;


  TParticle* particle
    = new(fParticles[fLoadPoint++]) 
      TParticle(pdg, is, parent, -1, kFirstDaughter, kLastDaughter,
		px, py, pz, e, vx, vy, vz, tof);
                
  particle->SetPolarisation(polx, poly, polz);
  particle->SetWeight(weight);
  particle->SetUniqueID(mech);

  
  
  if(!done) {
      particle->SetBit(kDoneBit);
  } else {
      particle->SetBit(kTransportBit);
      fNtransported++;
  }
  
  

  //  Declare that the daughter information is valid
  particle->SetBit(kDaughtersBit);
  //  Add the particle to the stack
  
  fParticleMap.AddAtAndExpand(particle, fNtrack);//CHECK!!

  if(parent>=0) {
      particle = GetParticleMapEntry(parent);
      if (particle) {
	  particle->SetLastDaughter(fNtrack);
	  if(particle->GetFirstDaughter()<0) particle->SetFirstDaughter(fNtrack);
      }
      else {
	  AliError(Form("Parent %d does not exist",parent));
      }
  } else { 
      //
      // This is a primary track. Set high water mark for this event
      fHgwmk = fNtrack;
      //
      // Set also number if primary tracks
      fNprimary = fHgwmk+1;
      fCurrentPrimary++;
  }
  ntr = fNtrack++;
}

//_____________________________________________________________________________
TParticle*  AliStack::PopNextTrack(Int_t& itrack)
{
  //
  // Returns next track from stack of particles
  //
  

  TParticle* track = GetNextParticle();

  if (track) {
    itrack = fCurrent;
    track->SetBit(kDoneBit);
  }
  else
    itrack = -1;
  
  fCurrentTrack = track;
  return track;
}

//_____________________________________________________________________________
TParticle*  AliStack::PopPrimaryForTracking(Int_t i)
{
  //
  // Returns i-th primary particle if it is flagged to be tracked,
  // 0 otherwise
  //
  
  TParticle* particle = Particle(i);
  
  if (!particle->TestBit(kDoneBit)) {
    fCurrentTrack = particle;
    return particle;
  }
  else
    return 0;
}      

//_____________________________________________________________________________
Bool_t AliStack::PurifyKine(Float_t rmax, Float_t zmax)
{
  //
  // Compress kinematic tree keeping only flagged particles
  // and renaming the particle id's in all the hits
  //

  int nkeep = fHgwmk + 1, parent, i;
  TParticle *part, *father;
  fTrackLabelMap.Set(fParticleMap.GetLast()+1);

  // Save in Header total number of tracks before compression
  // If no tracks generated return now
  if(fHgwmk+1 == fNtrack) return kFALSE;

  // First pass, invalid Daughter information
  for(i=0; i<fNtrack; i++) {
      // Preset map, to be removed later
      if(i<=fHgwmk) fTrackLabelMap[i]=i ; 
      else {
	  fTrackLabelMap[i] = -99;
	  if((part=GetParticleMapEntry(i))) {
//
//        Check of this track should be kept for physics reasons 
	    if (KeepPhysics(part, rmax, zmax)) KeepTrack(i);
//
	      part->ResetBit(kDaughtersBit);
	      part->SetFirstDaughter(-1);
	      part->SetLastDaughter(-1);
	  }
      }
  }
  // Invalid daughter information for the parent of the first particle
  // generated. This may or may not be the current primary according to
  // whether decays have been recorded among the primaries
  part = GetParticleMapEntry(fHgwmk+1);
  fParticleMap.At(part->GetFirstMother())->ResetBit(kDaughtersBit);
  // Second pass, build map between old and new numbering
  for(i=fHgwmk+1; i<fNtrack; i++) {
      if(fParticleMap.At(i)->TestBit(kKeepBit)) {
	  // This particle has to be kept
	  fTrackLabelMap[i]=nkeep;
	  // If old and new are different, have to move the pointer
	  if(i!=nkeep) fParticleMap[nkeep]=fParticleMap.At(i);
	  part = GetParticleMapEntry(nkeep);
	  // as the parent is always *before*, it must be already
	  // in place. This is what we are checking anyway!
	  if((parent=part->GetFirstMother())>fHgwmk) {
	      if(fTrackLabelMap[parent]==-99) Fatal("PurifyKine","fTrackLabelMap[%d] = -99!\n",parent);
	      else part->SetFirstMother(fTrackLabelMap[parent]);}
	  nkeep++;
      }
  }
  
  // Fix daughters information
  for (i=fHgwmk+1; i<nkeep; i++) {
      part = GetParticleMapEntry(i);
      parent = part->GetFirstMother();
      if(parent>=0) {
	  father = GetParticleMapEntry(parent);
	  if(father->TestBit(kDaughtersBit)) {
	      
	      if(i<father->GetFirstDaughter()) father->SetFirstDaughter(i);
	      if(i>father->GetLastDaughter())  father->SetLastDaughter(i);
	  } else {
	      // Initialise daughters info for first pass
	      father->SetFirstDaughter(i);
	      father->SetLastDaughter(i);
	      father->SetBit(kDaughtersBit);
	  }
      }
  }
  //
  // Now the output bit, from fHgwmk to nkeep we write everything and we erase
  if(nkeep > fParticleFileMap.GetSize()) fParticleFileMap.Set(Int_t (nkeep*1.5));
  for (i=fHgwmk+1; i<nkeep; ++i) {
      fParticleBuffer = GetParticleMapEntry(i);
      fParticleFileMap[i]=static_cast<Int_t>(TreeK()->GetEntries());
      TreeK()->Fill();
      fParticleMap[i]=fParticleBuffer=0;
  }
  
  for (i = nkeep; i < fNtrack; ++i) fParticleMap[i]=0;
  
  Int_t toshrink = fNtrack-fHgwmk-1;
  fLoadPoint-=toshrink;
  
  for(i=fLoadPoint; i<fLoadPoint+toshrink; ++i) fParticles.RemoveAt(i);
  fNtrack=nkeep;
  fHgwmk=nkeep-1;
  return kTRUE;
}


Bool_t AliStack::ReorderKine()
{
//
// In some transport code children might not come in a continuous sequence.
// In this case the stack  has  to  be reordered in order to establish the 
// mother daughter relation using index ranges.
//    
  if(fHgwmk+1 == fNtrack) return kFALSE;

  //
  // Howmany secondaries have been produced ?
  Int_t nNew = fNtrack - fHgwmk - 1;
    
  if (nNew > 0) {
      Int_t i, j;
      TArrayI map1(nNew);
      //
      // Copy pointers to temporary array
      TParticle** tmp = new TParticle*[nNew];
      
      for (i = 0; i < nNew; i++) {
	  if (fParticleMap.At(fHgwmk + 1 + i)) {
	      tmp[i] = GetParticleMapEntry(fHgwmk + 1 + i);
	  } else {
	      tmp[i] = 0x0;
	  }
	  map1[i] = -99;
      }
  
      
      //
      // Reset  LoadPoint 
      // 
      Int_t loadPoint = fHgwmk + 1;
      //
      // Re-Push particles into stack 
      // The outer loop is over parents, the inner over children.
      // -1 refers to the primary particle
      //
      for (i = -1; i < nNew-1; i++) {
	  Int_t ipa;
	  TParticle* parP;
	  if (i == -1) {
	      ipa  = tmp[0]->GetFirstMother();
	      parP = GetParticleMapEntry(ipa);
	  } else {
	      ipa = (fHgwmk + 1 + i);
              // Skip deleted particles
	      if (!tmp[i])                          continue;
              // Skip particles without children
	      if (tmp[i]->GetFirstDaughter() == -1) continue;
	      parP = tmp[i];
	  }
          // Reset daughter information

	  Int_t idaumin = parP->GetFirstDaughter() - fHgwmk - 1;
	  Int_t idaumax = parP->GetLastDaughter()  - fHgwmk - 1;
	  parP->SetFirstDaughter(-1);
	  parP->SetLastDaughter(-1);
	  for (j = idaumin; j <= idaumax; j++) {
              // Skip deleted particles
	      if (!tmp[j])        continue;
              // Skip particles already handled
	      if (map1[j] != -99) continue;
	      Int_t jpa = tmp[j]->GetFirstMother();
              // Check if daughter of current parent
	      if (jpa == ipa) {
		  fParticleMap[loadPoint] = tmp[j];
		  // Re-establish daughter information
		  parP->SetLastDaughter(loadPoint);
		  if (parP->GetFirstDaughter() == -1) parP->SetFirstDaughter(loadPoint);
		  // Set Mother information
		  if (i != -1) {
		      tmp[j]->SetFirstMother(map1[i]);
		  } 
		  // Build the map
		  map1[j] = loadPoint;
		  // Increase load point
		  loadPoint++;
	      }
	  } // children
      } // parents

      delete[] tmp;

      //
      // Build map for remapping of hits
      // 
      fTrackLabelMap.Set(fNtrack);
      for (i = 0; i < fNtrack; i ++) {
	  if (i <= fHgwmk) {
	      fTrackLabelMap[i] = i;
	  } else{
	      fTrackLabelMap[i] = map1[i - fHgwmk -1];
	  }
      }
  } // new particles poduced
  
  return kTRUE;
}

Bool_t AliStack::KeepPhysics(const TParticle* part, Float_t rmax, Float_t zmax)
{
    //
    // Some particles have to kept on the stack for reasons motivated
    // by physics analysis. Decision is put here.
    //
    Bool_t keep = kFALSE;



    Int_t parent = part->GetFirstMother();
    if (parent >= 0 && parent <= fHgwmk) {
      // Keep 1st generation secondaries in a pre-defined r-z range
      Float_t vx = part->Vx();
      Float_t vy = part->Vy();
      Float_t vz = part->Vz();
      Float_t r  = TMath::Sqrt(vx * vx + vy * vy);
      if (r < rmax && TMath::Abs(vz) < zmax) return kTRUE;
      //
      TParticle* father = GetParticleMapEntry(parent);
    //
    // Keep first-generation daughter from primaries with heavy flavor 
    //
	Int_t kf = father->GetPdgCode();
	kf = TMath::Abs(kf);
	Int_t kfl = kf;
	// meson ?
	if  (kfl > 10) kfl/=100;
	// baryon
	if (kfl > 10)  kfl/=10;
	if (kfl > 10)  kfl/=10;
	if (kfl >= 4) {
	    keep = kTRUE;
	}
	//
	// e+e- from pair production of primary gammas
	//
	if ((part->GetUniqueID()) == kPPair)  keep = kTRUE;
    }
    //
    // Decay(cascade) from primaries
    // 
    if ((part->GetUniqueID() == kPDecay) && (parent >= 0)) {
      // Particles from decay
      TParticle* father = GetParticleMapEntry(parent);
      Int_t imo = parent;
      while((imo > fHgwmk) && (father->GetUniqueID() == kPDecay)) {
	imo =  father->GetFirstMother();
	father = GetParticleMapEntry(imo);
      }
      if ((imo <= fHgwmk)) keep = kTRUE;
    }
    return keep;
}

//_____________________________________________________________________________
void AliStack::FinishEvent()
{
//
// Write out the kinematics that was not yet filled
//
  
// Update event header

  if (!TreeK()) {
//    Fatal("FinishEvent", "No kinematics tree is defined.");
//    Don't panic this is a probably a lego run
      return;
  }  
  
  CleanParents();
   if(TreeK()->GetEntries() ==0) {
    // set the fParticleFileMap size for the first time
    fParticleFileMap.Set(fHgwmk+1);
  }

  Bool_t allFilled = kFALSE;
  TParticle *part;
  for(Int_t i=0; i<fHgwmk+1; ++i) {
    if((part=GetParticleMapEntry(i))) {
      fParticleBuffer = part;
      fParticleFileMap[i]= static_cast<Int_t>(TreeK()->GetEntries());
      TreeK()->Fill();
      fParticleBuffer=0;      
      fParticleMap.AddAt(0,i);      
      
      // When all primaries were filled no particle!=0
      // should be left => to be removed later.
      if (allFilled) AliWarning(Form("Why != 0 part # %d?\n",i));
    }
    else 
    {
      // // printf("Why = 0 part # %d?\n",i); => We know.
      // break;
      // we don't break now in order to be sure there is no
      // particle !=0 left.
      // To be removed later and replaced with break.
       if(!allFilled) allFilled = kTRUE;
    }
  }
  AliInfoF("Ntrack=%d kept from %d transported\n",fNtrack,fNtransported);
} 
//_____________________________________________________________________________

void AliStack::FlagTrack(Int_t track)
{
  //
  // Flags a track and all its family tree to be kept
  //
  
  TParticle *particle;

  Int_t curr=track;
  while(1) {
    particle = GetParticleMapEntry(curr);
    
    // If the particle is flagged the three from here upward is saved already
    if(particle->TestBit(kKeepBit)) return;
    
    // Save this particle
    particle->SetBit(kKeepBit);
    
    // Move to father if any
    if((curr=particle->GetFirstMother())==-1) return;
  }
}
 
//_____________________________________________________________________________
void AliStack::KeepTrack(Int_t track)
{ 
  //
  // Flags a track to be kept
  //
  
  fParticleMap.At(track)->SetBit(kKeepBit);
}

//_____________________________________________________________________________
void  AliStack::Clean(Int_t size) 
{
  //
  // Reset stack data except for fTreeK
  //
  
  fNtrack=0;
  fNprimary=0;
  fNtransported=0;
  fHgwmk=0;
  fLoadPoint=0;
  fCurrent = -1;
  ResetArrays(size);
}

//_____________________________________________________________________________
void  AliStack::Reset(Int_t size) 
{
  //
  // Reset stack data including fTreeK
  //

  Clean(size);
  delete fParticleBuffer; fParticleBuffer = 0;
  fTreeK = 0x0;
}

//_____________________________________________________________________________
void  AliStack::ResetArrays(Int_t size) 
{
  //
  // Resets stack arrays
  //
  fParticles.Clear();
  fParticleMap.Clear();
  if (size>0) fParticleMap.Expand(size);
}

//_____________________________________________________________________________
void AliStack::SetHighWaterMark(Int_t)
{
  //
  // Set high water mark for last track in event
  //
    
    fHgwmk = fNtrack-1;
    fCurrentPrimary=fHgwmk;
    // Set also number of primary tracks
    fNprimary = fHgwmk+1;
}

//_____________________________________________________________________________
TParticle* AliStack::Particle(Int_t i, Bool_t useInEmbedding)
{
  //
  // Return particle with specified ID
  if (GetMCEmbeddingFlag() && !useInEmbedding) {
    AliError("Method should not be called by user in embedding mode, returning dummy particle");
    return GetDummyParticle();
  }
  if (i==gkDummyLabel) return 0;
  
  if(!fParticleMap.At(i)) {
    Int_t nentries = fParticles.GetEntriesFast();
    // algorithmic way of getting entry index
    // (primary particles are filled after secondaries)
    Int_t entry = TreeKEntry(i,useInEmbedding);
    // check whether algorithmic way and 
    // and the fParticleFileMap[i] give the same;
    // give the fatal error if not
    if (entry != fParticleFileMap[i]) {
      AliFatal(Form(
        "!! The algorithmic way and map are different: !!\n entry: %d map: %d",
	entry, fParticleFileMap[i])); 
    } 
    // Load particle at entry into fParticleBuffer
    TreeK()->GetEntry(entry);
    // Add to the TClonesarray
    new (fParticles[nentries]) TParticle(*fParticleBuffer);
    // Store a pointer in the TObjArray
    fParticleMap.AddAt(fParticles[nentries],i);
  }
  return GetParticleMapEntry(i);
}

//_____________________________________________________________________________
TParticle* AliStack::ParticleFromTreeK(Int_t id, Bool_t useInEmbedding) const
{
// 
// return pointer to TParticle with label id
//
  if (GetMCEmbeddingFlag() && !useInEmbedding) {
    AliError("Method should not be called by user in embedding mode, returning dummy particle");
    return GetDummyParticle();
  }
  Int_t entry;
  if ((entry = TreeKEntry(id,useInEmbedding)) < 0) return 0;
  if (fTreeK->GetEntry(entry)<=0) return 0;
  return fParticleBuffer;
}

//_____________________________________________________________________________
Int_t AliStack::TreeKEntry(Int_t id, Bool_t useInEmbedding) const 
{
//
// Return entry number in the TreeK for particle with label id
// Return negative number if label>fNtrack
//
// The order of particles in TreeK reflects the order of the transport of primaries and production of secondaries:
//
// Before transport there are fNprimary particles on the stack.
// They are transported one by one and secondaries (fNtrack - fNprimary) are produced. 
// After the transport of each particles secondaries are written to the TreeK
// They occupy the entries 0 ... fNtrack - fNprimary - 1
// The primaries are written after they have been transported and occupy 
// fNtrack - fNprimary .. fNtrack - 1

  if (GetMCEmbeddingFlag() && !useInEmbedding) {
    AliError("Method should not be called by user in embedding mode, returning -1");
    return -1;
  }

  
  Int_t entry;
  if (id<fNprimary)
    entry = id+fNtrack-fNprimary;
  else 
    entry = id-fNprimary;
  return entry;
}

//_____________________________________________________________________________
Int_t AliStack::GetCurrentParentTrackNumber() const
{
  //
  // Return number of the parent of the current track
  //
  
  TParticle* current = GetParticleMapEntry(fCurrent);

  if (current) 
    return current->GetFirstMother();
  else {
    AliWarning("Current track not found in the stack");
    return -1;
  }  
}
 
//_____________________________________________________________________________
Int_t AliStack::GetPrimary(Int_t id, Bool_t useInEmbedding)
{
  //
  // Return number of primary that has generated track
  //
  
  int current, parent;
  //
  parent=id;
  while (1) {
    current=parent;
    TParticle* part = Particle(current,useInEmbedding);
    if (!part || ( parent=part->GetFirstMother() )<0 ) return current;
  }
}
 
//_____________________________________________________________________________
void AliStack::DumpPart (Int_t i) const
{
  //
  // Dumps particle i in the stack
  //
  GetParticleMapEntry(i)->Print();
}

//_____________________________________________________________________________
void AliStack::DumpPStack ()
{
  //
  // Dumps the particle stack
  //

  Int_t i;

  printf("\n\n=======================================================================\n");
  for (i=0;i<fNtrack;i++) 
    {
      TParticle* particle = Particle(i);
      if (particle) {
        printf("-> %d ",i); particle->Print();
        printf("--------------------------------------------------------------\n");
      }
      else 
        Warning("DumpPStack", "No particle with id %d.", i); 
    }	 

  printf("\n=======================================================================\n\n");
  
  // print  particle file map
  // printf("\nParticle file map: \n");
  // for (i=0; i<fNtrack; i++) 
  //     printf("   %d th entry: %d \n",i,fParticleFileMap[i]);
}


//_____________________________________________________________________________
void AliStack::DumpLoadedStack() const
{
  //
  // Dumps the particle in the stack
  // that are loaded in memory.
  //

  printf(
	 "\n\n=======================================================================\n");
  for (Int_t i=0;i<fNtrack;i++) 
    {
      TParticle* particle = GetParticleMapEntry(i);
      if (particle) {
        printf("-> %d ",i); particle->Print();
        printf("--------------------------------------------------------------\n");
      }
      else { 	
        printf("-> %d  Particle not loaded.\n",i);
        printf("--------------------------------------------------------------\n");
      }	
    }
  printf(
	 "\n=======================================================================\n\n");
}

//_____________________________________________________________________________
void  AliStack::SetCurrentTrack(Int_t track)
{ 
  fCurrent = track; 
  if (fCurrent < fNprimary) fCurrentTrack = Particle(track);
}


//_____________________________________________________________________________
//
// protected methods
//

//_____________________________________________________________________________
void AliStack::CleanParents()
{
  //
  // Clean particles stack
  // Set parent/daughter relations
  //
  
  TParticle *part;
  int i;
  for(i=0; i<fHgwmk+1; i++) {
    part = GetParticleMapEntry(i);
    if(part) if(!part->TestBit(kDaughtersBit)) {
      part->SetFirstDaughter(-1);
      part->SetLastDaughter(-1);
    }
  }
}

//_____________________________________________________________________________
TParticle* AliStack::GetNextParticle()
{
  //
  // Return next particle from stack of particles
  //
  
  TParticle* particle = 0;
  
  // search secondaries
  //for(Int_t i=fNtrack-1; i>=0; i--) {
  for(Int_t i=fNtrack-1; i>fHgwmk; i--) {
      particle = GetParticleMapEntry(i);
      if ((particle) && (!particle->TestBit(kDoneBit))) {
	  fCurrent=i;    
	  return particle;
      }   
  }    

  // take next primary if all secondaries were done
  while (fCurrentPrimary>=0) {
      fCurrent = fCurrentPrimary;    
      particle = GetParticleMapEntry(fCurrentPrimary--);
      if ((particle) && (!particle->TestBit(kDoneBit))) {
	  return particle;
      } 
  }
  
  // nothing to be tracked
  fCurrent = -1;
 
  
  return particle;  
}
//__________________________________________________________________________________________

void AliStack::ConnectTree(TTree* tree)
{
//
//  Creates branch for writing particles
//

  fTreeK = tree;
    
  AliDebug(1, "Connecting TreeK");
  if (fTreeK == 0x0)
   {
    if (TreeK() == 0x0)
     {
      AliFatal("Parameter is NULL");//we don't like such a jokes
      return;
     }
    return;//in this case TreeK() calls back this method (ConnectTree) 
           //tree after setting fTreeK, the rest was already executed
           //it is safe to return now
   }

 //  Create a branch for particles   
  
  AliDebug(2, Form("Tree name is %s",fTreeK->GetName()));
   
  if (fTreeK->GetDirectory())
   {
     AliDebug(2, Form("and dir is %s",fTreeK->GetDirectory()->GetName()));
   }    
  else
    AliWarning("DIR IS NOT SET !!!");
  
  TBranch *branch=fTreeK->GetBranch("Particles");
  if(branch == 0x0)
   {
    branch = fTreeK->Branch("Particles", &fParticleBuffer, 4000);
    AliDebug(2, "Creating Branch in Tree");
   }  
  else
   {
    AliDebug(2, "Branch Found in Tree");
    branch->SetAddress(&fParticleBuffer);
   }
  if (branch->GetDirectory())
   {
    AliDebug(1, Form("Branch Dir Name is %s",branch->GetDirectory()->GetName()));
   } 
  else
    AliWarning("Branch Dir is NOT SET");
}

//_____________________________________________________________________________

Bool_t AliStack::GetEvent()
{
//
// Get new event from TreeK

    // Reset/Create the particle stack
    Int_t size = (Int_t)TreeK()->GetEntries();
    ResetArrays(size);
    return kTRUE;
}
//_____________________________________________________________________________

Bool_t AliStack::IsStable(Int_t pdg) const
{
  //
  // Decide whether particle (pdg) is stable
  //
  
  
  // All ions/nucleons are considered as stable
  // Nuclear code is 10LZZZAAAI
  if(pdg>1000000000)return kTRUE;

  const Int_t kNstable = 18;
  Int_t i;
  
  Int_t pdgStable[kNstable] = {
    kGamma,             // Photon
    kElectron,          // Electron
    kMuonPlus,          // Muon 
    kPiPlus,            // Pion
    kKPlus,             // Kaon
    kK0Short,           // K0s
    kK0Long,            // K0l
    kProton,            // Proton 
    kNeutron,           // Neutron
    kLambda0,           // Lambda_0
    kSigmaMinus,        // Sigma Minus
    kSigmaPlus,         // Sigma Plus
    3312,               // Xsi Minus 
    3322,               // Xsi 
    3334,               // Omega
    kNuE,               // Electron Neutrino 
    kNuMu,              // Muon Neutrino
    kNuTau              // Tau Neutrino
  };
    
  Bool_t isStable = kFALSE;
  for (i = 0; i < kNstable; i++) {
    if (pdg == TMath::Abs(pdgStable[i])) {
      isStable = kTRUE;
      break;
    }
  }
  
  return isStable;
}

//_____________________________________________________________________________
Bool_t AliStack::IsPhysicalPrimary(Int_t index, Bool_t useInEmbedding)
{
    //
    // Test if a particle is a physical primary according to the following definition:
    // Particles produced in the collision including products of strong and
    // electromagnetic decay and excluding feed-down from weak decays of strange
    // particles.
    //
    TParticle* p = Particle(index,useInEmbedding);
    if (!p) return kFALSE;
    Int_t ist = p->GetStatusCode();
    Int_t pdg = TMath::Abs(p->GetPdgCode());    
    //
    // Initial state particle
    // Solution for K0L decayed by Pythia6
    // ->
    if ((ist > 1) && (pdg!=130) && index < GetNprimary()) return kFALSE;
    if ((ist > 1) && index >= GetNprimary()) return kFALSE;
    // <-

    
    if (!IsStable(pdg)) return kFALSE;
    if (index < GetNprimary()) {
//
// Particle produced by generator
      // Solution for K0L decayed by Pythia6
      // ->
      Int_t ipm =  p->GetFirstMother();
      if (ipm > -1) {
	TParticle* ppm  = Particle(ipm, useInEmbedding);
	if (TMath::Abs(ppm->GetPdgCode()) == 130) return kFALSE;
      }
      // <-
	return kTRUE;
    } else {
//
// Particle produced during transport
//

	Int_t imo =  p->GetFirstMother();
	TParticle* pm  = Particle(imo,useInEmbedding);
	Int_t mpdg = TMath::Abs(pm->GetPdgCode());
// Check for Sigma0 
	if ((mpdg == 3212) &&  (imo <  GetNprimary())) return kTRUE;
// 
// Check if it comes from a pi0 decay
//
	if ((mpdg == kPi0) && (imo < GetNprimary()))   return kTRUE; 

// Check if this is a heavy flavor decay product
	Int_t mfl  = Int_t (mpdg / TMath::Power(10, Int_t(TMath::Log10(mpdg))));
	//
	// Light hadron
	if (mfl < 4) return kFALSE;
	
	//
	// Heavy flavor hadron produced by generator
	if (imo <  GetNprimary()) {
	    return kTRUE;
	}
	
	// To be sure that heavy flavor has not been produced in a secondary interaction
	// Loop back to the generated mother
	while (imo >=  GetNprimary()) {
	    imo = pm->GetFirstMother();
	    pm  =  Particle(imo,useInEmbedding);
	}
	mpdg = TMath::Abs(pm->GetPdgCode());
	mfl  = Int_t (mpdg / TMath::Power(10, Int_t(TMath::Log10(mpdg))));

	if (mfl < 4) {
	    return kFALSE;
	} else {
	    return kTRUE;
	} 
    } // produced by generator ?
} 

Bool_t AliStack::IsSecondaryFromWeakDecay(Int_t index, Bool_t useInEmbedding) {

  // If a particle is not a physical primary, check if it comes from weak decay

  if(IsPhysicalPrimary(index,useInEmbedding)) return kFALSE;
  
  TParticle* particle = Particle(index, useInEmbedding);
  if (!particle) return kFALSE;
  Int_t uniqueID = particle->GetUniqueID();

  Int_t indexMoth = particle->GetFirstMother();
  if(indexMoth < 0) return kFALSE; // if index mother < 0 and not a physical primary, is a non-stable product or one of the beams
  TParticle* moth = Particle(indexMoth,useInEmbedding);
  Float_t codemoth = (Float_t)TMath::Abs(moth->GetPdgCode());
  // mass of the flavour
  Int_t mfl = 0;
  // Protect the "rootino" case when codemoth is 0
  if (TMath::Abs(codemoth)>0) mfl = Int_t (codemoth / TMath::Power(10, Int_t(TMath::Log10(codemoth))));
  
  if(mfl == 3 && uniqueID == kPDecay) return kTRUE;// The first mother is strange and it's a decay
  if(codemoth == 211 && uniqueID == kPDecay) return kTRUE;// pion+- decay products
  if(codemoth == 13 && uniqueID == kPDecay) return kTRUE;// muon decay products

  /// Hypernuclei case
  if (TMath::Abs(moth->GetPdgCode()) > 1000000000 && uniqueID == kPDecay) {
    if ((moth->GetPdgCode() / 10000000) % 10 != 0) return kTRUE; /// Number of lambdas in the hypernucleus != 0
  }

  return kFALSE;
  
}
Bool_t AliStack::IsSecondaryFromMaterial(Int_t index, Bool_t useInEmbedding) {

  // If a particle is not a physical primary, check if it comes from material

  if(IsPhysicalPrimary(index,useInEmbedding)) return kFALSE;
  if(IsSecondaryFromWeakDecay(index,useInEmbedding)) return kFALSE;
  TParticle* particle = Particle(index,useInEmbedding);
  if (!particle) return kFALSE;
  Int_t indexMoth = particle->GetFirstMother();
  if(indexMoth < 0) return kFALSE; // if index mother < 0 and not a physical primary, is a non-stable product or one of the beams
  return kTRUE;

}

//__________________________________________
TParticle* AliStack::GetDummyParticle()
{
  static TParticle dummy(21,999,-1,-1,-1,-1,1,1,999,999,0,0,0,0);
  return &dummy;
}
