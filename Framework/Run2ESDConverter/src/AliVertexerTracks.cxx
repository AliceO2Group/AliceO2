/**************************************************************************
 * Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
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

//-----------------------------------------------------------------
//    Implementation of the vertexer from tracks
//
// Origin: AliITSVertexerTracks
//         A.Dainese, Padova, 
//         andrea.dainese@pd.infn.it
//         M.Masera,  Torino, 
//         massimo.masera@to.infn.it
// Moved to STEER and adapted to ESD tracks: 
//         F.Prino,  Torino, prino@to.infn.it
//-----------------------------------------------------------------

//---- Root headers --------
#include <TSystem.h>
#include <TClonesArray.h>
#include <TDirectory.h>
#include <TFile.h>
//---- AliRoot headers -----
#include "AliStrLine.h"
#include "AliExternalTrackParam.h"
#include "AliNeutralTrackParam.h"
#include "AliVEvent.h"
#include "AliVTrack.h"
#include "AliESDtrack.h"
#include "AliESDEvent.h"
#include "AliVertexerTracks.h"


ClassImp(AliVertexerTracks)


//----------------------------------------------------------------------------
AliVertexerTracks::AliVertexerTracks():
TObject(),
fVert(),
fCurrentVertex(0),
fMode(0),
fFieldkG(-999.),
fTrkArraySel(),
fIdSel(0),
fTrksToSkip(0),
fNTrksToSkip(0),
fConstraint(kFALSE),
fOnlyFitter(kFALSE),
fMinTracks(1),
fMinClusters(3),
fDCAcut(0.1),
fDCAcutIter0(0.1),
fNSigma(3.),
fMaxd0z0(0.5),
fMinDetFitter(100.),
fMaxTgl(1000.),
fITSrefit(kTRUE),
fITSpureSA(kFALSE),
fFiducialR(3.),
fFiducialZ(30.),
fnSigmaForUi00(1.5),
fAlgo(1),
fAlgoIter0(4),
fSelectOnTOFBunchCrossing(kFALSE), 
fKeepAlsoUnflaggedTOFBunchCrossing(kTRUE),
fMVWSum(0),
fMVWE2(0),
fMVTukey2(6.),
fMVSigma2(1.),
fMVSig2Ini(1.e3),
fMVMaxSigma2(3.),
fMVMinSig2Red(0.005),
fMVMinDst(10.e-4),
fMVScanStep(3.),
fMVMaxWghNtr(10.),
fMVFinalWBinary(kTRUE),
fBCSpacing(50),
fMVVertices(0),
fDisableBCInCPass0(kTRUE),
fClusterize(kFALSE),
fDeltaZCutForCluster(0.1),
fnSigmaZCutForCluster(999999.)
{
//
// Default constructor
//
  SetVtxStart();
  SetVtxStartSigma();
}
//----------------------------------------------------------------------------
AliVertexerTracks::AliVertexerTracks(Double_t fieldkG):
TObject(),
fVert(),
fCurrentVertex(0),
fMode(0),
fFieldkG(fieldkG),
fTrkArraySel(),
fIdSel(0),
fTrksToSkip(0),
fNTrksToSkip(0),
fConstraint(kFALSE),
fOnlyFitter(kFALSE),
fMinTracks(1),
fMinClusters(3),
fDCAcut(0.1),
fDCAcutIter0(0.1),
fNSigma(3.),
fMaxd0z0(0.5),
fMinDetFitter(100.),
fMaxTgl(1000.),
fITSrefit(kTRUE),
fITSpureSA(kFALSE),
fFiducialR(3.),
fFiducialZ(30.),
fnSigmaForUi00(1.5),
fAlgo(1),
fAlgoIter0(4),
fSelectOnTOFBunchCrossing(kFALSE), 
fKeepAlsoUnflaggedTOFBunchCrossing(kTRUE),
fMVWSum(0),
fMVWE2(0),
fMVTukey2(6.),
fMVSigma2(1.),
fMVSig2Ini(1.e3),
fMVMaxSigma2(3.),
fMVMinSig2Red(0.005),
fMVMinDst(10.e-4),
fMVScanStep(3.),
fMVMaxWghNtr(10.),
fMVFinalWBinary(kTRUE),
fBCSpacing(50),
fMVVertices(0),
fDisableBCInCPass0(kTRUE),
fClusterize(kFALSE),
fDeltaZCutForCluster(0.1),
fnSigmaZCutForCluster(999999.)
{
//
// Standard constructor
//
  SetVtxStart();
  SetVtxStartSigma();
}
//-----------------------------------------------------------------------------
AliVertexerTracks::~AliVertexerTracks() 
{
  // Default Destructor
  // The objects pointed by the following pointer are not owned
  // by this class and are not deleted
  fCurrentVertex = 0;
  if(fTrksToSkip) { delete [] fTrksToSkip; fTrksToSkip=NULL; fNTrksToSkip=0;}
  if(fIdSel) { delete [] fIdSel; fIdSel=NULL; }
  if(fMVVertices) delete fMVVertices;
}

//----------------------------------------------------------------------------
AliESDVertex* AliVertexerTracks::FindPrimaryVertex(const AliVEvent *vEvent)
{
//
// Primary vertex for current ESD or AOD event
// (Two iterations: 
//  1st with 5*fNSigma*sigma cut w.r.t. to initial vertex
//      + cut on sqrt(d0d0+z0z0) if fConstraint=kFALSE  
//  2nd with fNSigma*sigma cut w.r.t. to vertex found in 1st iteration) 
//
  fCurrentVertex = 0;
  TString evtype = vEvent->IsA()->GetName();
  Bool_t inputAOD = ((evtype=="AliAODEvent") ? kTRUE : kFALSE);

  if(inputAOD && fMode==1) {
    printf("Error : AliVertexerTracks: no TPC-only vertex from AOD\n"); 
    TooFewTracks(); 
    return fCurrentVertex;
  }

  // accept 1-track case only if constraint is available
  if(!fConstraint && fMinTracks==1) fMinTracks=2;

  // read tracks from AlivEvent
  Int_t nTrks = (Int_t)vEvent->GetNumberOfTracks();
  if(nTrks<fMinTracks) {
    TooFewTracks();
    return fCurrentVertex;
  } 
  //
  int bcRound = fBCSpacing/25;   // profit from larger than 25ns spacing and set correct BC
  //TDirectory * olddir = gDirectory;
  //  TFile *f = 0;
  //  if(nTrks>500) f = new TFile("VertexerTracks.root","recreate");
  TObjArray trkArrayOrig(nTrks);
  UShort_t *idOrig = new UShort_t[nTrks];
  Double_t *zTr = new Double_t[nTrks];
  Double_t *err2zTr = new Double_t[nTrks];

  Int_t nTrksOrig=0;
  AliExternalTrackParam *t=0;
  // loop on tracks
  for(Int_t i=0; i<nTrks; i++) {
    AliVTrack *track = (AliVTrack*)vEvent->GetTrack(i);
    // check tracks to skip
    Bool_t skipThis = kFALSE;
    for(Int_t j=0; j<fNTrksToSkip; j++) { 
      if(track->GetID()==fTrksToSkip[j]) {
	AliDebug(1,Form("skipping track: %d",i));
	skipThis = kTRUE;
      }
    }
    if(skipThis) continue;

    // skip pure ITS SA tracks (default)
    if(!fITSpureSA && (track->GetStatus()&AliESDtrack::kITSpureSA)) continue;
    // or use only pure ITS SA tracks
    if(fITSpureSA && !(track->GetStatus()&AliESDtrack::kITSpureSA)) continue;

    // kITSrefit
    if(fMode==0 && fITSrefit && !(track->GetStatus()&AliESDtrack::kITSrefit)) continue;

    if(!inputAOD) {  // ESD
      AliESDtrack* esdt = (AliESDtrack*)track;
      if(esdt->GetNcls(fMode) < fMinClusters) continue;
      if(fMode==0) {        // ITS mode
	Double_t x,p[5],cov[15];
	esdt->GetExternalParameters(x,p);
	esdt->GetExternalCovariance(cov);
	t = new AliExternalTrackParam(x,esdt->GetAlpha(),p,cov);
      } else if(fMode==1) { // TPC mode
	t = (AliExternalTrackParam*)esdt->GetTPCInnerParam();
	if(!t) continue;
	Double_t radius = 2.8; //something less than the beam pipe radius
	if(!PropagateTrackTo(t,radius)) continue;
      }
    } else {          // AOD (only ITS mode)
      if(track->GetID()<0) continue; // exclude global constrained and TPC only tracks (filter bits 128 and 512)
      Int_t ncls0=0;
      for(Int_t l=0;l<6;l++) if(TESTBIT(track->GetITSClusterMap(),l)) ncls0++;
      if(ncls0 < fMinClusters) continue;
      t = new AliExternalTrackParam(); t->CopyFromVTrack(track);
    }

    // use TOF info about bunch crossing
    if(fSelectOnTOFBunchCrossing) {
      double tdiff = track->GetTOFExpTDiff(fFieldkG);
      int bc = TMath::Nint(tdiff/25);
      // use only values with good margin
      if (bc<=AliVTrack::kTOFBCNA || TMath::Abs(tdiff/25.-bc)>0.4) bc = AliVTrack::kTOFBCNA;
      else bc /= bcRound;
      t->SetUniqueID(UInt_t(bc + kTOFBCShift));
    }
    //
    trkArrayOrig.AddLast(t);
    idOrig[nTrksOrig]=(UShort_t)track->GetID();
    zTr[nTrksOrig]=t->GetZ();
    err2zTr[nTrksOrig]=t->GetSigmaZ2();

    nTrksOrig++;
  } // end loop on tracks
  
  // call method that will reconstruct the vertex
  if(fClusterize) FindAllVertices(nTrksOrig,&trkArrayOrig,zTr,err2zTr,idOrig);
  else FindPrimaryVertex(&trkArrayOrig,idOrig);
  if(!inputAOD) AnalyzePileUp((AliESDEvent*)vEvent);

  if(fMode==0) trkArrayOrig.Delete();
  delete [] idOrig; idOrig=NULL;
  delete [] zTr; zTr=NULL;
  delete [] err2zTr; err2zTr=NULL;

  /*
  if(f) {
    f->Close(); delete f; f = NULL;
    gSystem->Unlink("VertexerTracks.root");
    olddir->cd();
  }
  */
  // set vertex ID for tracks used in the fit
  // (only for ESD)
  if(!inputAOD && fCurrentVertex) {
    Int_t nIndices = fCurrentVertex->GetNIndices();
    UShort_t *indices = fCurrentVertex->GetIndices();
    for(Int_t ind=0; ind<nIndices; ind++) {
      AliESDtrack *esdt = (AliESDtrack*)vEvent->GetTrack(indices[ind]);
      esdt->SetVertexID(-1);
    }
  }
 
  return fCurrentVertex;
}
//----------------------------------------------------------------------------
AliESDVertex* AliVertexerTracks::FindPrimaryVertex(const TObjArray *trkArrayOrig,
						   UShort_t *idOrig)
{
//
// Primary vertex using the AliExternalTrackParam's in the TObjArray.
// idOrig must contain the track IDs from AliESDtrack::GetID()
// (Two iterations: 
//  1st with 5*fNSigma*sigma cut w.r.t. to initial vertex
//      + cut on sqrt(d0d0+z0z0) if fConstraint=kFALSE  
//  2nd with fNSigma*sigma cut w.r.t. to vertex found in 1st iteration) 
//
  fCurrentVertex = 0;
  // accept 1-track case only if constraint is available
  if(!fConstraint && fMinTracks==1) fMinTracks=2;

  // read tracks from array
  Int_t nTrksOrig = (Int_t)trkArrayOrig->GetEntriesFast();
  AliDebug(1,Form("Initial number of tracks: %d",nTrksOrig));
  if(nTrksOrig<fMinTracks) {
    AliDebug(1,"TooFewTracks");
    TooFewTracks();
    return fCurrentVertex;
  } 

  // If fConstraint=kFALSE
  // run VertexFinder(1) to get rough estimate of initVertex (x,y)
  if(!fConstraint) {
    // fill fTrkArraySel, for VertexFinder()
    fIdSel = new UShort_t[nTrksOrig];
    PrepareTracks(*trkArrayOrig,idOrig,0);
    if(fIdSel) { delete [] fIdSel; fIdSel=NULL; }
    Double_t cutsave = fDCAcut;  
    fDCAcut = fDCAcutIter0;
    // vertex finder
    switch (fAlgoIter0) {
    case 1: StrLinVertexFinderMinDist(1); break;
    case 2: StrLinVertexFinderMinDist(0); break;
    case 3: HelixVertexFinder();          break;
    case 4: VertexFinder(1);              break;
    case 5: VertexFinder(0);              break;
    default: {AliFatal(Form("Wrong seeder algorithm %d",fAlgoIter0));} break;  
    }
    fDCAcut = cutsave;
    if(fVert.GetNContributors()>0) {
      fVert.GetXYZ(fNominalPos);
      fNominalPos[0] = fVert.GetX();
      fNominalPos[1] = fVert.GetY();
      fNominalPos[2] = fVert.GetZ();
      AliDebug(1,Form("No mean vertex: VertexFinder gives (%f, %f, %f)",fNominalPos[0],fNominalPos[1],fNominalPos[2]));
    } else {
      fNominalPos[0] = 0.;
      fNominalPos[1] = 0.;
      fNominalPos[2] = 0.;
      AliDebug(1,"No mean vertex and VertexFinder failed");
    }
  }
  
  // TWO ITERATIONS:
  //
  // ITERATION 1
  // propagate tracks to fNominalPos vertex
  // preselect them:
  // if(constraint) reject for |d0|>5*fNSigma*sigma w.r.t. fNominal... vertex
  // else  reject for |d0|\oplus|z0| > 5 mm w.r.t. fNominal... vertex
  // ITERATION 2
  // propagate tracks to best between initVertex and fCurrentVertex
  // preselect tracks (reject for |d0|>fNSigma*sigma w.r.t. best 
  //                   between initVertex and fCurrentVertex) 
  Bool_t multiMode = kFALSE;
  for(Int_t iter=1; iter<=2; iter++) {
    if (fAlgo==kMultiVertexer && iter==2) break; // multivertexer does not need 2 iterations
    if(fOnlyFitter && iter==1) continue; 
    if(fIdSel) { delete [] fIdSel; fIdSel=NULL; }
    fIdSel = new UShort_t[nTrksOrig];
    Int_t nTrksSel = PrepareTracks(*trkArrayOrig,idOrig,iter);
    AliDebug(1,Form("N tracks selected in iteration %d: %d",iter,nTrksSel));
    if(nTrksSel < fMinTracks) {
      TooFewTracks();
      return fCurrentVertex; 
    }

    // vertex finder
    if(!fOnlyFitter) {
      if(nTrksSel==1 && fAlgo!=kMultiVertexer) {
	AliDebug(1,"Just one track");
	OneTrackVertFinder();
      } else {
	switch (fAlgo) {
        case kStrLinVertexFinderMinDist1: StrLinVertexFinderMinDist(1); break;
        case kStrLinVertexFinderMinDist0: StrLinVertexFinderMinDist(0); break;
        case kHelixVertexFinder         : HelixVertexFinder();          break;
        case kVertexFinder1             : VertexFinder(1);              break;
        case kVertexFinder0             : VertexFinder(0);              break;
	case kMultiVertexer             : FindVerticesMV(); multiMode = kTRUE; break;
        default: {AliFatal(Form("Wrong vertexer algorithm %d",fAlgo));} break;  
	}
      }
      AliDebug(1," Vertex finding completed");
    }
    if (multiMode) break; // // multivertexer does not need 2nd iteration
    // vertex fitter
    VertexFitter();
  } // end loop on the two iterations

  if (!multiMode || fMVVertices->GetEntries()==0) { // in multi-vertex mode this is already done for found vertices
    // set indices of used tracks
    UShort_t *indices = 0;
    if(fCurrentVertex->GetNContributors()>0) {
      Int_t nIndices = (Int_t)fTrkArraySel.GetEntriesFast();
      indices = new UShort_t[nIndices];
      for(Int_t jj=0; jj<nIndices; jj++)
	indices[jj] = fIdSel[jj];
      fCurrentVertex->SetIndices(nIndices,indices);
    }
    if (indices) {delete [] indices; indices=NULL;}
    //
    // set vertex title
    TString title="VertexerTracksNoConstraint";
    if(fConstraint) {
      title="VertexerTracksWithConstraint";
      if(fOnlyFitter) title.Append("OnlyFitter");
    }
    fCurrentVertex->SetTitle(title.Data());
    //
    AliDebug(1,Form("xyz: %f %f %f; nc %d",fCurrentVertex->GetX(),fCurrentVertex->GetY(),fCurrentVertex->GetZ(),fCurrentVertex->GetNContributors()));
  }
  // clean up
  delete [] fIdSel; fIdSel=NULL;
  fTrkArraySel.Delete();
  if(fTrksToSkip) { delete [] fTrksToSkip; fTrksToSkip=NULL; }
  //
  
  return fCurrentVertex;
}
//------------------------------------------------------------------------
Double_t AliVertexerTracks::GetDeterminant3X3(Double_t matr[][3])
{
  //
  Double_t det=matr[0][0]*matr[1][1]*matr[2][2]-matr[0][0]*matr[1][2]*matr[2][1]-matr[0][1]*matr[1][0]*matr[2][2]+matr[0][1]*matr[1][2]*matr[2][0]+matr[0][2]*matr[1][0]*matr[2][1]-matr[0][2]*matr[1][1]*matr[2][0];
 return det;
}
//-------------------------------------------------------------------------
void AliVertexerTracks::GetStrLinDerivMatrix(const Double_t *p0,const Double_t *p1,Double_t (*m)[3],Double_t *d)
{
  //
  Double_t x12=p0[0]-p1[0];
  Double_t y12=p0[1]-p1[1];
  Double_t z12=p0[2]-p1[2];
  Double_t kk=x12*x12+y12*y12+z12*z12;
  m[0][0]=2-2/kk*x12*x12;
  m[0][1]=-2/kk*x12*y12;
  m[0][2]=-2/kk*x12*z12;
  m[1][0]=-2/kk*x12*y12;
  m[1][1]=2-2/kk*y12*y12;
  m[1][2]=-2/kk*y12*z12;
  m[2][0]=-2/kk*x12*z12;
  m[2][1]=-2/kk*y12*z12;
  m[2][2]=2-2/kk*z12*z12;
  d[0]=2*p0[0]-2/kk*p0[0]*x12*x12-2/kk*p0[2]*x12*z12-2/kk*p0[1]*x12*y12;
  d[1]=2*p0[1]-2/kk*p0[1]*y12*y12-2/kk*p0[0]*x12*y12-2/kk*p0[2]*z12*y12;
  d[2]=2*p0[2]-2/kk*p0[2]*z12*z12-2/kk*p0[0]*x12*z12-2/kk*p0[1]*z12*y12;

}
//--------------------------------------------------------------------------  
void AliVertexerTracks::GetStrLinDerivMatrix(const Double_t *p0,const Double_t *p1,const Double_t *sigmasq,Double_t (*m)[3],Double_t *d)
{
  //
  Double_t x12=p1[0]-p0[0];
  Double_t y12=p1[1]-p0[1];
  Double_t z12=p1[2]-p0[2];

  Double_t den= x12*x12*sigmasq[1]*sigmasq[2]+y12*y12*sigmasq[0]*sigmasq[2]+z12*z12*sigmasq[0]*sigmasq[1];

  Double_t kk= 2*(x12*x12/sigmasq[0]+y12*y12/sigmasq[1]+z12*z12/sigmasq[2]);

  Double_t cc[3];
  cc[0]=-x12/sigmasq[0];
  cc[1]=-y12/sigmasq[1];
  cc[2]=-z12/sigmasq[2];

  Double_t ww=(-p0[0]*x12*sigmasq[1]*sigmasq[2]-p0[1]*y12*sigmasq[0]*sigmasq[2]-p0[2]*z12*sigmasq[0]*sigmasq[1])/den;

  Double_t ss= -p0[0]*cc[0]-p0[1]*cc[1]-p0[2]*cc[2];

  Double_t aa[3];
  aa[0]=x12*sigmasq[1]*sigmasq[2]/den;
  aa[1]=y12*sigmasq[0]*sigmasq[2]/den;
  aa[2]=z12*sigmasq[0]*sigmasq[1]/den;

  m[0][0]=aa[0]*(aa[0]*kk+2*cc[0])+2*cc[0]*aa[0]+2/sigmasq[0];
  m[0][1]=aa[1]*(aa[0]*kk+2*cc[0])+2*cc[1]*aa[0];
  m[0][2]=aa[2]*(aa[0]*kk+2*cc[0])+2*cc[2]*aa[0];

  m[1][0]=aa[0]*(aa[1]*kk+2*cc[1])+2*cc[0]*aa[1];
  m[1][1]=aa[1]*(aa[1]*kk+2*cc[1])+2*cc[1]*aa[1]+2/sigmasq[1];
  m[1][2]=aa[2]*(aa[1]*kk+2*cc[1])+2*cc[2]*aa[1];

  m[2][0]=aa[0]*(aa[2]*kk+2*cc[2])+2*cc[0]*aa[2];
  m[2][1]=aa[1]*(aa[2]*kk+2*cc[2])+2*cc[1]*aa[2];
  m[2][2]=aa[2]*(aa[2]*kk+2*cc[2])+2*cc[2]*aa[2]+2/sigmasq[2];

  d[0]=-ww*(aa[0]*kk+2*cc[0])-2*ss*aa[0]+2*p0[0]/sigmasq[0];
  d[1]=-ww*(aa[1]*kk+2*cc[1])-2*ss*aa[1]+2*p0[1]/sigmasq[1];
  d[2]=-ww*(aa[2]*kk+2*cc[2])-2*ss*aa[2]+2*p0[2]/sigmasq[2];

  }
//--------------------------------------------------------------------------   
Double_t AliVertexerTracks::GetStrLinMinDist(const Double_t *p0,const Double_t *p1,const Double_t *x0)
{
  //
  Double_t x12=p0[0]-p1[0];
  Double_t y12=p0[1]-p1[1];
  Double_t z12=p0[2]-p1[2];
  Double_t x10=p0[0]-x0[0];
  Double_t y10=p0[1]-x0[1];
  Double_t z10=p0[2]-x0[2];
  //  return ((x10*x10+y10*y10+z10*z10)*(x12*x12+y12*y12+z12*z12)-(x10*x12+y10*y12+z10*z12)*(x10*x12+y10*y12+z10*z12))/(x12*x12+y12*y12+z12*z12);
  return ((y10*z12-z10*y12)*(y10*z12-z10*y12)+
	  (z10*x12-x10*z12)*(z10*x12-x10*z12)+
	  (x10*y12-y10*x12)*(x10*y12-y10*x12))
    /(x12*x12+y12*y12+z12*z12);
}
//---------------------------------------------------------------------------
void AliVertexerTracks::OneTrackVertFinder() 
{
  // find vertex for events with 1 track, using DCA to nominal beam axis
  AliDebug(1,Form("Number of prepared tracks =%d - Call OneTrackVertFinder",fTrkArraySel.GetEntries()));
  AliExternalTrackParam *track1;
  track1 = (AliExternalTrackParam*)fTrkArraySel.At(0);
  Double_t alpha=track1->GetAlpha();
  Double_t mindist = TMath::Cos(alpha)*fNominalPos[0]+TMath::Sin(alpha)*fNominalPos[1];
  Double_t pos[3],dir[3]; 
  track1->GetXYZAt(mindist,GetFieldkG(),pos);
  track1->GetPxPyPzAt(mindist,GetFieldkG(),dir);
  AliStrLine *line1 = new AliStrLine(pos,dir);
  Double_t p1[3]={fNominalPos[0],fNominalPos[1],0.}; 
  Double_t p2[3]={fNominalPos[0],fNominalPos[1],10.}; 
  AliStrLine *zeta=new AliStrLine(p1,p2,kTRUE);
  Double_t crosspoint[3]={0.,0.,0.};
  Double_t sigma=999.;
  Int_t nContrib=-1;
  Int_t retcode = zeta->Cross(line1,crosspoint);
  if(retcode>=0){
    sigma=line1->GetDistFromPoint(crosspoint);
    nContrib=1;
  }
  delete zeta;
  delete line1;
  fVert.SetXYZ(crosspoint);
  fVert.SetDispersion(sigma);
  fVert.SetNContributors(nContrib);  
}
//---------------------------------------------------------------------------
void AliVertexerTracks::HelixVertexFinder() 
{
  // Get estimate of vertex position in (x,y) from tracks DCA


  Double_t initPos[3];
  initPos[2] = 0.;
  for(Int_t i=0;i<2;i++)initPos[i]=fNominalPos[i];

  Int_t nacc = (Int_t)fTrkArraySel.GetEntriesFast();

  Double_t aver[3]={0.,0.,0.};
  Double_t averquad[3]={0.,0.,0.};
  Double_t sigmaquad[3]={0.,0.,0.};
  Double_t sigma=0;
  Int_t ncombi = 0;
  AliExternalTrackParam *track1;
  AliExternalTrackParam *track2;
  Double_t distCA;
  Double_t x;
  Double_t alpha, cs, sn;
  Double_t crosspoint[3];
  for(Int_t i=0; i<nacc; i++){
    track1 = (AliExternalTrackParam*)fTrkArraySel.At(i);
    

    for(Int_t j=i+1; j<nacc; j++){
      track2 = (AliExternalTrackParam*)fTrkArraySel.At(j);

      distCA=track2->PropagateToDCA(track1,GetFieldkG());
      if(fDCAcut<=0 ||(fDCAcut>0&&distCA<fDCAcut)){
	x=track1->GetX();
	alpha=track1->GetAlpha();
	cs=TMath::Cos(alpha); sn=TMath::Sin(alpha);
	Double_t x1=x*cs - track1->GetY()*sn;
	Double_t y1=x*sn + track1->GetY()*cs;
	Double_t z1=track1->GetZ();
	
	Double_t sx1=sn*sn*track1->GetSigmaY2(), sy1=cs*cs*track1->GetSigmaY2(); 
	x=track2->GetX();
	alpha=track2->GetAlpha();
	cs=TMath::Cos(alpha); sn=TMath::Sin(alpha);
	Double_t x2=x*cs - track2->GetY()*sn;
	Double_t y2=x*sn + track2->GetY()*cs;
	Double_t z2=track2->GetZ();
	Double_t sx2=sn*sn*track2->GetSigmaY2(), sy2=cs*cs*track2->GetSigmaY2();
	Double_t sz1=track1->GetSigmaZ2(), sz2=track2->GetSigmaZ2();
	Double_t wx1=sx2/(sx1+sx2), wx2=1.- wx1;
	Double_t wy1=sy2/(sy1+sy2), wy2=1.- wy1;
	Double_t wz1=sz2/(sz1+sz2), wz2=1.- wz1;
	crosspoint[0]=wx1*x1 + wx2*x2; 
	crosspoint[1]=wy1*y1 + wy2*y2; 
	crosspoint[2]=wz1*z1 + wz2*z2;

	ncombi++;
	for(Int_t jj=0;jj<3;jj++)aver[jj]+=crosspoint[jj];
	for(Int_t jj=0;jj<3;jj++)averquad[jj]+=(crosspoint[jj]*crosspoint[jj]);
      }
    }
      
  }
  if(ncombi>0){
    for(Int_t jj=0;jj<3;jj++){
      initPos[jj] = aver[jj]/ncombi;
      averquad[jj]/=ncombi;
      sigmaquad[jj]=averquad[jj]-initPos[jj]*initPos[jj];
      sigma+=sigmaquad[jj];
    }
    sigma=TMath::Sqrt(TMath::Abs(sigma));
  }
  else {
    Warning("HelixVertexFinder","Finder did not succed");
    sigma=999;
  }
  fVert.SetXYZ(initPos);
  fVert.SetDispersion(sigma);
  fVert.SetNContributors(ncombi);
}
//----------------------------------------------------------------------------
Int_t AliVertexerTracks::PrepareTracks(const TObjArray &trkArrayOrig,
				       const UShort_t *idOrig,
				       Int_t optImpParCut) 
{
//
// Propagate tracks to initial vertex position and store them in a TObjArray
//
  AliDebug(1," PrepareTracks()");

  Int_t nTrksOrig = (Int_t)trkArrayOrig.GetEntriesFast();
  Int_t nTrksSel = 0;
  Double_t maxd0rphi; 
  Double_t sigmaCurr[3];
  Double_t normdistx,normdisty;
  Double_t d0z0[2],covd0z0[3]; 
  Double_t sigmad0;

  AliESDVertex *initVertex = new AliESDVertex(fNominalPos,fNominalCov,1,1);

  if(!fTrkArraySel.IsEmpty()) fTrkArraySel.Delete();

  AliExternalTrackParam *track=0;

  // loop on tracks
  for(Int_t i=0; i<nTrksOrig; i++) {
    AliExternalTrackParam *trackOrig=(AliExternalTrackParam*)trkArrayOrig.At(i);
    if(trackOrig->Charge()!=0) { // normal tracks
      track = new AliExternalTrackParam(*(AliExternalTrackParam*)trkArrayOrig.At(i));
    } else { // neutral tracks (from a V0)
      track = new AliNeutralTrackParam(*(AliNeutralTrackParam*)trkArrayOrig.At(i));
    }
    track->SetUniqueID(trackOrig->GetUniqueID());
    // tgl cut
    if(TMath::Abs(track->GetTgl())>fMaxTgl) {
      AliDebug(1,Form(" rejecting track with tgl = %f",track->GetTgl()));
      delete track; continue;
    }

    Bool_t propagateOK = kFALSE, cutond0z0 = kTRUE;
    // propagate track to vertex
    if(optImpParCut<2 || fOnlyFitter) { // optImpParCut==1 or 0
      propagateOK = track->PropagateToDCA(initVertex,GetFieldkG(),100.,d0z0,covd0z0);
    } else {              // optImpParCut==2
      fCurrentVertex->GetSigmaXYZ(sigmaCurr);
      normdistx = TMath::Abs(fCurrentVertex->GetX()-fNominalPos[0])/TMath::Sqrt(sigmaCurr[0]*sigmaCurr[0]+fNominalCov[0]);
      normdisty = TMath::Abs(fCurrentVertex->GetY()-fNominalPos[1])/TMath::Sqrt(sigmaCurr[1]*sigmaCurr[1]+fNominalCov[2]);
      AliDebug(1,Form("normdistx %f  %f    %f",fCurrentVertex->GetX(),fNominalPos[0],TMath::Sqrt(sigmaCurr[0]*sigmaCurr[0]+fNominalCov[0])));
      AliDebug(1,Form("normdisty %f  %f    %f",fCurrentVertex->GetY(),fNominalPos[1],TMath::Sqrt(sigmaCurr[1]*sigmaCurr[1]+fNominalCov[2])));
      AliDebug(1,Form("sigmaCurr %f %f    %f",sigmaCurr[0],sigmaCurr[1],TMath::Sqrt(fNominalCov[0])+TMath::Sqrt(fNominalCov[2])));
      if(normdistx < 3. && normdisty < 3. &&
	 (sigmaCurr[0]+sigmaCurr[1])<(TMath::Sqrt(fNominalCov[0])+TMath::Sqrt(fNominalCov[2]))) {
	propagateOK = track->PropagateToDCA(fCurrentVertex,GetFieldkG(),100.,d0z0,covd0z0);
      } else {
	propagateOK = track->PropagateToDCA(initVertex,GetFieldkG(),100.,d0z0,covd0z0);
	if(fConstraint) cutond0z0=kFALSE;
      }
    }

    if(!propagateOK) { 
      AliDebug(1,"     rejected");
      delete track; continue; 
    }

    sigmad0 = TMath::Sqrt(covd0z0[0]);
    maxd0rphi = fNSigma*sigmad0;
    if(optImpParCut==1) maxd0rphi *= 5.;
    maxd0rphi = TMath::Min(maxd0rphi,fFiducialR); 
    //sigmad0z0 = TMath::Sqrt(covd0z0[0]+covd0z0[2]);

    AliDebug(1,Form("trk %d; id %d; |d0| = %f;  d0 cut = %f; |z0| = %f; |d0|oplus|z0| = %f; d0z0 cut = %f",i,(Int_t)idOrig[i],TMath::Abs(d0z0[0]),maxd0rphi,TMath::Abs(d0z0[1]),TMath::Sqrt(d0z0[0]*d0z0[0]+d0z0[1]*d0z0[1]),fMaxd0z0));


    //---- track selection based on impact parameters ----//

    // always reject tracks outside fiducial volume
    if(TMath::Abs(d0z0[0])>fFiducialR || TMath::Abs(d0z0[1])>fFiducialZ) { 
      AliDebug(1,"     rejected");
      delete track; continue; 
    }

    // during iterations 1 and 2 , reject tracks with d0rphi > maxd0rphi
    if(optImpParCut>0 && TMath::Abs(d0z0[0])>maxd0rphi) { 
      AliDebug(1,"     rejected");
      delete track; continue; 
    }

    // if fConstraint=kFALSE, during iterations 1 and 2 ||
    // if fConstraint=kTRUE, during iteration 2,
    // select tracks with d0oplusz0 < fMaxd0z0
    if((!fConstraint && optImpParCut>0 && fVert.GetNContributors()>0) ||
       ( fConstraint && optImpParCut==2 && cutond0z0)) {
      if(nTrksOrig>=3 && 
	 TMath::Sqrt(d0z0[0]*d0z0[0]+d0z0[1]*d0z0[1])>fMaxd0z0) { 
	AliDebug(1,"     rejected");
	delete track; continue; 
      }
    }
    
    // track passed all selections
    fTrkArraySel.AddLast(track);
    fIdSel[nTrksSel] = idOrig[i];
    nTrksSel++; 
  } // end loop on tracks

  delete initVertex;

  return nTrksSel;
} 
//----------------------------------------------------------------------------
Bool_t AliVertexerTracks::PropagateTrackTo(AliExternalTrackParam *track,
					   Double_t xToGo) {
  //----------------------------------------------------------------
  // COPIED from AliTracker
  //
  // Propagates the track to the plane X=xk (cm).
  //
  //  Origin: Marian Ivanov,  Marian.Ivanov@cern.ch
  //----------------------------------------------------------------

  const Double_t kEpsilon = 0.00001;
  Double_t xpos = track->GetX();
  Double_t dir = (xpos<xToGo) ? 1. : -1.;
  Double_t maxStep = 5;
  Double_t maxSnp = 0.8;
  //
  while ( (xToGo-xpos)*dir > kEpsilon){
    Double_t step = dir*TMath::Min(TMath::Abs(xToGo-xpos), maxStep);
    Double_t x    = xpos+step;
    Double_t xyz0[3],xyz1[3];
    track->GetXYZ(xyz0);   //starting global position

    if(!track->GetXYZAt(x,GetFieldkG(),xyz1)) return kFALSE;   // no prolongation
    xyz1[2]+=kEpsilon; // waiting for bug correction in geo

    if(TMath::Abs(track->GetSnpAt(x,GetFieldkG())) >= maxSnp) return kFALSE;
    if(!track->PropagateTo(x,GetFieldkG()))  return kFALSE;

    if(TMath::Abs(track->GetSnp()) >= maxSnp) return kFALSE;
    track->GetXYZ(xyz0);   // global position
    Double_t alphan = TMath::ATan2(xyz0[1], xyz0[0]); 
    //
    Double_t ca=TMath::Cos(alphan-track->GetAlpha()), 
      sa=TMath::Sin(alphan-track->GetAlpha());
    Double_t sf=track->GetSnp(), cf=TMath::Sqrt((1.-sf)*(1.+sf));
    Double_t sinNew =  sf*ca - cf*sa;
    if(TMath::Abs(sinNew) >= maxSnp) return kFALSE;
    if(!track->Rotate(alphan)) return kFALSE;
 
    xpos = track->GetX();
  }
  return kTRUE;
}
//---------------------------------------------------------------------------
AliESDVertex* AliVertexerTracks::RemoveTracksFromVertex(AliESDVertex *inVtx,
							const TObjArray *trkArray,
							UShort_t *id,
							const Float_t *diamondxy) const
{
//
// Removes tracks in trksTree from fit of inVtx
//

  if(!strstr(inVtx->GetTitle(),"VertexerTracksWithConstraint")) {
    printf("ERROR: primary vertex has no constraint: cannot remove tracks");
    return 0x0;
  }
  if(!strstr(inVtx->GetTitle(),"VertexerTracksWithConstraintOnlyFitter"))
    printf("WARNING: result of tracks' removal will be only approximately correct");

  TMatrixD rv(3,1);
  rv(0,0) = inVtx->GetX();
  rv(1,0) = inVtx->GetY();
  rv(2,0) = inVtx->GetZ();
  TMatrixD vV(3,3);
  Double_t cov[6];
  inVtx->GetCovMatrix(cov);
  vV(0,0) = cov[0];
  vV(0,1) = cov[1]; vV(1,0) = cov[1];
  vV(1,1) = cov[2];
  vV(0,2) = cov[3]; vV(2,0) = cov[3];
  vV(1,2) = cov[4]; vV(2,1) = cov[4]; 
  vV(2,2) = cov[5];

  TMatrixD sumWi(TMatrixD::kInverted,vV);
  TMatrixD sumWiri(sumWi,TMatrixD::kMult,rv);

  Int_t nUsedTrks = inVtx->GetNIndices();
  Double_t chi2 = inVtx->GetChi2();

  AliExternalTrackParam *track = 0;
  Int_t ntrks = (Int_t)trkArray->GetEntriesFast();
  for(Int_t i=0;i<ntrks;i++) {
    track = (AliExternalTrackParam*)trkArray->At(i);
    if(!inVtx->UsesTrack(id[i])) {
      printf("track %d was not used in vertex fit",id[i]);
      continue;
    }
    Double_t alpha = track->GetAlpha();
    Double_t xl = diamondxy[0]*TMath::Cos(alpha)+diamondxy[1]*TMath::Sin(alpha);
    track->PropagateTo(xl,GetFieldkG()); 
    // vector of track global coordinates
    TMatrixD ri(3,1);
    // covariance matrix of ri
    TMatrixD wWi(3,3);
    
    // get space point from track
    if(!TrackToPoint(track,ri,wWi)) continue;

    TMatrixD wWiri(wWi,TMatrixD::kMult,ri); 

    sumWi -= wWi;
    sumWiri -= wWiri;

    // track contribution to chi2
    TMatrixD deltar = rv; deltar -= ri;
    TMatrixD wWideltar(wWi,TMatrixD::kMult,deltar);
    Double_t chi2i = deltar(0,0)*wWideltar(0,0)+
                     deltar(1,0)*wWideltar(1,0)+
	             deltar(2,0)*wWideltar(2,0);
    // remove from total chi2
    chi2 -= chi2i;

    nUsedTrks--;
    if(nUsedTrks<2) {
      printf("Trying to remove too many tracks!");
      return 0x0;
    }
  }

  TMatrixD rvnew(3,1);
  TMatrixD vVnew(3,3);

  // new inverted of weights matrix
  TMatrixD invsumWi(TMatrixD::kInverted,sumWi);
  vVnew = invsumWi;
  // new position of primary vertex
  rvnew.Mult(vVnew,sumWiri);

  Double_t position[3];
  position[0] = rvnew(0,0);
  position[1] = rvnew(1,0);
  position[2] = rvnew(2,0);
  cov[0] = vVnew(0,0);
  cov[1] = vVnew(0,1);
  cov[2] = vVnew(1,1);
  cov[3] = vVnew(0,2);
  cov[4] = vVnew(1,2);
  cov[5] = vVnew(2,2);
  
  // store data in the vertex object
  AliESDVertex *outVtx = new AliESDVertex(position,cov,chi2,nUsedTrks+1); // the +1 is for the constraint
  outVtx->SetTitle(inVtx->GetTitle());
  UShort_t *inindices = inVtx->GetIndices();
  Int_t nIndices = nUsedTrks;
  UShort_t *outindices = new UShort_t[nIndices];
  Int_t j=0;
  for(Int_t k=0; k<inVtx->GetNIndices(); k++) {
    Bool_t copyindex=kTRUE;
    for(Int_t l=0; l<ntrks; l++) {
      if(inindices[k]==id[l]) {copyindex=kFALSE; break;}
    }
    if(copyindex) {
      outindices[j] = inindices[k]; j++;
    }
  }
  outVtx->SetIndices(nIndices,outindices);
  if (outindices) delete [] outindices;

  /*
    printf("Vertex before removing tracks:");
    inVtx->PrintStatus();
    inVtx->PrintIndices();
    printf("Vertex after removing tracks:");
    outVtx->PrintStatus();
    outVtx->PrintIndices();
  */

  return outVtx;
}
//---------------------------------------------------------------------------
AliESDVertex* AliVertexerTracks::RemoveConstraintFromVertex(AliESDVertex *inVtx,
						     Float_t *diamondxyz,
						     Float_t *diamondcov) const
{
//
// Removes diamond constraint from fit of inVtx
//

  if(!strstr(inVtx->GetTitle(),"VertexerTracksWithConstraint")) {
    printf("ERROR: primary vertex has no constraint: cannot remove it\n");
    return 0x0;
  }
  if(inVtx->GetNContributors()<3) {
    printf("ERROR: primary vertex has less than 2 tracks: cannot remove contraint\n");
    return 0x0;
  }

  // diamond constraint 
  TMatrixD vVb(3,3);
  vVb(0,0) = diamondcov[0];
  vVb(0,1) = diamondcov[1];
  vVb(0,2) = 0.;
  vVb(1,0) = diamondcov[1];
  vVb(1,1) = diamondcov[2];
  vVb(1,2) = 0.;
  vVb(2,0) = 0.;
  vVb(2,1) = 0.;
  vVb(2,2) = diamondcov[5];
  TMatrixD vVbInv(TMatrixD::kInverted,vVb);
  TMatrixD rb(3,1);
  rb(0,0) = diamondxyz[0];
  rb(1,0) = diamondxyz[1];
  rb(2,0) = diamondxyz[2];
  TMatrixD vVbInvrb(vVbInv,TMatrixD::kMult,rb);

  // input vertex
  TMatrixD rv(3,1);
  rv(0,0) = inVtx->GetX();
  rv(1,0) = inVtx->GetY();
  rv(2,0) = inVtx->GetZ();
  TMatrixD vV(3,3);
  Double_t cov[6];
  inVtx->GetCovMatrix(cov);
  vV(0,0) = cov[0];
  vV(0,1) = cov[1]; vV(1,0) = cov[1];
  vV(1,1) = cov[2];
  vV(0,2) = cov[3]; vV(2,0) = cov[3];
  vV(1,2) = cov[4]; vV(2,1) = cov[4]; 
  vV(2,2) = cov[5];
  TMatrixD vVInv(TMatrixD::kInverted,vV);
  TMatrixD vVInvrv(vVInv,TMatrixD::kMult,rv);


  TMatrixD sumWi = vVInv - vVbInv;


  TMatrixD sumWiri = vVInvrv - vVbInvrb;

  TMatrixD rvnew(3,1);
  TMatrixD vVnew(3,3);

  // new inverted of weights matrix
  TMatrixD invsumWi(TMatrixD::kInverted,sumWi);
  vVnew = invsumWi;
  // new position of primary vertex
  rvnew.Mult(vVnew,sumWiri);

  Double_t position[3];
  position[0] = rvnew(0,0);
  position[1] = rvnew(1,0);
  position[2] = rvnew(2,0);
  cov[0] = vVnew(0,0);
  cov[1] = vVnew(0,1);
  cov[2] = vVnew(1,1);
  cov[3] = vVnew(0,2);
  cov[4] = vVnew(1,2);
  cov[5] = vVnew(2,2);


  Double_t chi2 = inVtx->GetChi2();

  // diamond constribution to chi2
  TMatrixD deltar = rv; deltar -= rb;
  TMatrixD vVbInvdeltar(vVbInv,TMatrixD::kMult,deltar);
  Double_t chi2b = deltar(0,0)*vVbInvdeltar(0,0)+
                   deltar(1,0)*vVbInvdeltar(1,0)+
                   deltar(2,0)*vVbInvdeltar(2,0);
  // remove from total chi2
  chi2 -= chi2b;

  // store data in the vertex object
  AliESDVertex *outVtx = new AliESDVertex(position,cov,chi2,inVtx->GetNContributors()-1);
  outVtx->SetTitle("VertexerTracksNoConstraint");
  UShort_t *inindices = inVtx->GetIndices();
  Int_t nIndices = inVtx->GetNIndices();
  outVtx->SetIndices(nIndices,inindices);

  return outVtx;
}
//---------------------------------------------------------------------------
void AliVertexerTracks::SetCuts(Double_t *cuts, Int_t ncuts) 
{
//
//  Cut values
//
  if (ncuts>0) SetDCAcut(cuts[0]);
  if (ncuts>1) SetDCAcutIter0(cuts[1]);
  if (ncuts>2) SetMaxd0z0(cuts[2]);
  if (ncuts>3) if(fMode==0 && cuts[3]<0) SetITSrefitNotRequired();
  if (ncuts>3) SetMinClusters((Int_t)(TMath::Abs(cuts[3])));
  if (ncuts>4) SetMinTracks((Int_t)(cuts[4]));
  if (ncuts>5) SetNSigmad0(cuts[5]);
  if (ncuts>6) SetMinDetFitter(cuts[6]);
  if (ncuts>7) SetMaxTgl(cuts[7]);
  if (ncuts>9) SetFiducialRZ(cuts[8],cuts[9]);
  if (ncuts>10) fAlgo=(Int_t)(cuts[10]);
  if (ncuts>11) fAlgoIter0=(Int_t)(cuts[11]);
  //
  if (ncuts>12) if (cuts[12]>1.)   SetMVTukey2(cuts[12]);
  if (ncuts>13) if (cuts[13]>1.)   SetMVSig2Ini(cuts[13]);
  if (ncuts>14) if (cuts[14]>0.1)  SetMVMaxSigma2(cuts[14]);
  if (ncuts>15) if (cuts[15]>1e-5) SetMVMinSig2Red(cuts[15]);
  if (ncuts>16) if (cuts[16]>1e-5) SetMVMinDst(cuts[16]);
  if (ncuts>17) if (cuts[17]>0.5)  SetMVScanStep(cuts[17]);
  if (ncuts>18) SetMVMaxWghNtr(cuts[18]);
  if (ncuts>19) SetMVFinalWBinary(cuts[19]>0);
  if (ncuts>20) SetBCSpacing(int(cuts[20]));
  //
  if (ncuts>21) if (cuts[21]>0.5)  SetUseTrackClusterization(kTRUE);
  if (ncuts>22) SetDeltaZCutForCluster(cuts[22]);
  if (ncuts>23) SetnSigmaZCutForCluster(cuts[23]);  
  //
  if ( (fAlgo==kMultiVertexer || fClusterize) && fBCSpacing>0) SetSelectOnTOFBunchCrossing(kTRUE,kTRUE);
  else                       SetSelectOnTOFBunchCrossing(kFALSE,kTRUE);
  //
  // Don't use BCSpacing in CPass0
  TString cpass = gSystem->Getenv("CPass");
  if (cpass=="0" && fDisableBCInCPass0) {
    AliInfoF("CPass%s declared, switch off using BC from TOF",cpass.Data());
    SetBCSpacing(-25);
    SetSelectOnTOFBunchCrossing(kFALSE,kTRUE);
  }

  return;
}
//---------------------------------------------------------------------------
void AliVertexerTracks::SetITSMode(Double_t dcacut,
				   Double_t dcacutIter0,
				   Double_t maxd0z0,
				   Int_t minCls,
				   Int_t mintrks,
				   Double_t nsigma,
				   Double_t mindetfitter,
				   Double_t maxtgl,
				   Double_t fidR,
				   Double_t fidZ,
				   Int_t finderAlgo,
				   Int_t finderAlgoIter0)
{
//
//  Cut values for ITS mode
//
  fMode = 0;
  if(minCls>0) {
    SetITSrefitRequired();
  } else {
    SetITSrefitNotRequired();
  }
  SetDCAcut(dcacut);
  SetDCAcutIter0(dcacutIter0);
  SetMaxd0z0(maxd0z0);
  SetMinClusters(TMath::Abs(minCls));
  SetMinTracks(mintrks);
  SetNSigmad0(nsigma);
  SetMinDetFitter(mindetfitter);
  SetMaxTgl(maxtgl);
  SetFiducialRZ(fidR,fidZ);
  fAlgo=finderAlgo;
  fAlgoIter0=finderAlgoIter0;

  return; 
}
//---------------------------------------------------------------------------
void AliVertexerTracks::SetTPCMode(Double_t dcacut,
				   Double_t dcacutIter0,
				   Double_t maxd0z0,
				   Int_t minCls,
				   Int_t mintrks,
				   Double_t nsigma,
				   Double_t mindetfitter,
				   Double_t maxtgl,
				   Double_t fidR,
				   Double_t fidZ,
				   Int_t finderAlgo,
				   Int_t finderAlgoIter0) 
{
//
//  Cut values for TPC mode
//
  fMode = 1;
  SetITSrefitNotRequired();
  SetDCAcut(dcacut);
  SetDCAcutIter0(dcacutIter0);
  SetMaxd0z0(maxd0z0);
  SetMinClusters(minCls);
  SetMinTracks(mintrks);
  SetNSigmad0(nsigma);
  SetMinDetFitter(mindetfitter);
  SetMaxTgl(maxtgl);
  SetFiducialRZ(fidR,fidZ);
  fAlgo=finderAlgo;
  fAlgoIter0=finderAlgoIter0;

  return; 
}
//---------------------------------------------------------------------------
void AliVertexerTracks::SetSkipTracks(Int_t n,const Int_t *skipped) 
{
//
// Mark the tracks not to be used in the vertex reconstruction.
// Tracks are identified by AliESDtrack::GetID()
//
  delete[] fTrksToSkip;
  fNTrksToSkip = n;  fTrksToSkip = new Int_t[n]; 
  for(Int_t i=0;i<n;i++) fTrksToSkip[i] = skipped[i]; 
  return; 
}
//---------------------------------------------------------------------------
void  AliVertexerTracks::SetVtxStart(AliESDVertex *vtx) 
{ 
//
// Set initial vertex knowledge
//
  vtx->GetXYZ(fNominalPos);
  vtx->GetCovMatrix(fNominalCov);
  SetConstraintOn();
  return; 
}
//---------------------------------------------------------------------------
void AliVertexerTracks::StrLinVertexFinderMinDist(Int_t optUseWeights)
{
  AliExternalTrackParam *track1;
  const Int_t knacc = (Int_t)fTrkArraySel.GetEntriesFast();
  AliStrLine **linarray = new AliStrLine* [knacc];
  for(Int_t i=0; i<knacc; i++){
    track1 = (AliExternalTrackParam*)fTrkArraySel.At(i);
    Double_t alpha=track1->GetAlpha();
    Double_t mindist = TMath::Cos(alpha)*fNominalPos[0]+TMath::Sin(alpha)*fNominalPos[1];
    Double_t pos[3],dir[3],sigmasq[3]; 
    track1->GetXYZAt(mindist,GetFieldkG(),pos);
    track1->GetPxPyPzAt(mindist,GetFieldkG(),dir);
    sigmasq[0]=TMath::Sin(alpha)*TMath::Sin(alpha)*track1->GetSigmaY2();
    sigmasq[1]=TMath::Cos(alpha)*TMath::Cos(alpha)*track1->GetSigmaY2();
    sigmasq[2]=track1->GetSigmaZ2();
    TMatrixD ri(3,1);
    TMatrixD wWi(3,3);
    if(!TrackToPoint(track1,ri,wWi)) {
      optUseWeights=kFALSE;
      AliDebug(1,"WARNING: TrackToPoint failed");
    }
    Double_t wmat[9];
    Int_t iel=0;
    for(Int_t ia=0;ia<3;ia++){
      for(Int_t ib=0;ib<3;ib++){
	wmat[iel]=wWi(ia,ib);
	iel++;
      }    
    }
    linarray[i] = new AliStrLine(pos,sigmasq,wmat,dir);
  }
  fVert=TrackletVertexFinder(linarray,knacc,optUseWeights);
  for(Int_t i=0; i<knacc; i++) delete linarray[i];
  delete [] linarray;
}
//---------------------------------------------------------------------------
AliESDVertex AliVertexerTracks::TrackletVertexFinder(const TClonesArray *lines, Int_t optUseWeights)
{
  // Calculate the point at minimum distance to prepared tracks (TClonesArray)
  const Int_t knacc = (Int_t)lines->GetEntriesFast();
  AliStrLine** lines2 = new AliStrLine* [knacc];
  for(Int_t i=0; i<knacc; i++){
    lines2[i]= (AliStrLine*)lines->At(i);
  }
  AliESDVertex vert = TrackletVertexFinder(lines2,knacc,optUseWeights); 
  delete [] lines2;
  return vert;
}

//---------------------------------------------------------------------------
AliESDVertex AliVertexerTracks::TrackletVertexFinder(AliStrLine **lines, const Int_t knacc, Int_t optUseWeights)
{
  // Calculate the point at minimum distance to prepared tracks (array of AliStrLine) 

  Double_t initPos[3]={0.,0.,0.};

  Double_t (*vectP0)[3]=new Double_t [knacc][3];
  Double_t (*vectP1)[3]=new Double_t [knacc][3];
  
  Double_t sum[3][3];
  Double_t dsum[3]={0,0,0};
  TMatrixD sumWi(3,3);
  for(Int_t i=0;i<3;i++){
    for(Int_t j=0;j<3;j++){
      sum[i][j]=0;
      sumWi(i,j)=0.;
    }
  }

  for(Int_t i=0; i<knacc; i++){
    AliStrLine *line1 = lines[i]; 
    Double_t p0[3],cd[3],sigmasq[3];
    Double_t wmat[9];
    if(!line1) {printf("ERROR %d %d\n",i,knacc); continue;}
    line1->GetP0(p0);
    line1->GetCd(cd);
    line1->GetSigma2P0(sigmasq);
    line1->GetWMatrix(wmat);
    TMatrixD wWi(3,3);
    Int_t iel=0;
    for(Int_t ia=0;ia<3;ia++){
      for(Int_t ib=0;ib<3;ib++){
	wWi(ia,ib)=wmat[iel];
	iel++;
      }    
    }

    sumWi+=wWi;

    Double_t p1[3]={p0[0]+cd[0],p0[1]+cd[1],p0[2]+cd[2]};
    vectP0[i][0]=p0[0];
    vectP0[i][1]=p0[1];
    vectP0[i][2]=p0[2];
    vectP1[i][0]=p1[0];
    vectP1[i][1]=p1[1];
    vectP1[i][2]=p1[2];
    
    Double_t matr[3][3];
    Double_t dknow[3];
    if(optUseWeights==0)GetStrLinDerivMatrix(p0,p1,matr,dknow);
    else GetStrLinDerivMatrix(p0,p1,sigmasq,matr,dknow);


    for(Int_t iii=0;iii<3;iii++){
      dsum[iii]+=dknow[iii]; 
      for(Int_t lj=0;lj<3;lj++) sum[iii][lj]+=matr[iii][lj];
    }
  }
 
  TMatrixD invsumWi(TMatrixD::kInverted,sumWi);
  Double_t covmatrix[6];
  covmatrix[0] = invsumWi(0,0);
  covmatrix[1] = invsumWi(0,1);
  covmatrix[2] = invsumWi(1,1);
  covmatrix[3] = invsumWi(0,2);
  covmatrix[4] = invsumWi(1,2);
  covmatrix[5] = invsumWi(2,2);

  Double_t vett[3][3];
  Double_t det=GetDeterminant3X3(sum);
  Double_t sigma=0;
  
  if(TMath::Abs(det) > kAlmost0){
    for(Int_t zz=0;zz<3;zz++){
      for(Int_t ww=0;ww<3;ww++){
	for(Int_t kk=0;kk<3;kk++) vett[ww][kk]=sum[ww][kk];
      }
      for(Int_t kk=0;kk<3;kk++) vett[kk][zz]=dsum[kk];
      initPos[zz]=GetDeterminant3X3(vett)/det;
    }


    for(Int_t i=0; i<knacc; i++){
      Double_t p0[3]={0,0,0},p1[3]={0,0,0};
      for(Int_t ii=0;ii<3;ii++){
	p0[ii]=vectP0[i][ii];
	p1[ii]=vectP1[i][ii];
      }
      sigma+=GetStrLinMinDist(p0,p1,initPos);
    }

    if(sigma>0.) {sigma=TMath::Sqrt(sigma);}else{sigma=999;}
  }else{
    sigma=999;
  }
  AliESDVertex theVert(initPos,covmatrix,99999.,knacc);
  theVert.SetDispersion(sigma);
  delete [] vectP0;
  delete [] vectP1;
  return theVert;
}

//---------------------------------------------------------------------------
Bool_t AliVertexerTracks::TrackToPoint(AliExternalTrackParam *t,
				       TMatrixD &ri,TMatrixD &wWi,
				       Bool_t uUi3by3) const 
{
//
// Extract from the AliExternalTrackParam the global coordinates ri and covariance matrix
// wWi of the space point that it represents (to be used in VertexFitter())
//

  
  Double_t rotAngle = t->GetAlpha();
  if(rotAngle<0.) rotAngle += 2.*TMath::Pi();
  Double_t cosRot = TMath::Cos(rotAngle);
  Double_t sinRot = TMath::Sin(rotAngle);
  /*
  // RS >>>
  Double_t lambda = TMath::ATan(t->GetTgl());
  Double_t cosLam   = TMath::Cos(lambda);
  Double_t sinLam   = TMath::Sin(lambda);
  // RS <<<
  */
  ri(0,0) = t->GetX()*cosRot-t->GetY()*sinRot;
  ri(1,0) = t->GetX()*sinRot+t->GetY()*cosRot;
  ri(2,0) = t->GetZ();

  if(!uUi3by3) {
    // matrix to go from global (x,y,z) to local (y,z);
    TMatrixD qQi(2,3);
    qQi(0,0) = -sinRot;
    qQi(0,1) = cosRot;
    qQi(0,2) = 0.;
    //
    qQi(1,0) = 0.;
    qQi(1,1) = 0.;
    qQi(1,2) = 1.;
    //
    // RS: Added polar inclination
    /*
    qQi(1,0) = -sinLam*cosRot;
    qQi(1,1) = -sinLam*sinRot;
    qQi(1,2) = cosLam;
    */
    // covariance matrix of local (y,z) - inverted
    TMatrixD uUi(2,2);
    uUi(0,0) = t->GetSigmaY2();
    uUi(0,1) = t->GetSigmaZY();
    uUi(1,0) = t->GetSigmaZY();
    uUi(1,1) = t->GetSigmaZ2();
    //printf(" Ui :");
    //printf(" %f   %f",uUi(0,0),uUi(0,1));
    //printf(" %f   %f",uUi(1,0),uUi(1,1));

    if(uUi.Determinant() <= 0.) return kFALSE;
    TMatrixD uUiInv(TMatrixD::kInverted,uUi);
  
    // weights matrix: wWi = qQiT * uUiInv * qQi
    TMatrixD uUiInvQi(uUiInv,TMatrixD::kMult,qQi);
    TMatrixD m(qQi,TMatrixD::kTransposeMult,uUiInvQi);
    wWi = m;
  } else {
    if(fVert.GetNContributors()<1) AliFatal("Vertex from finder is empty");
    // matrix to go from global (x,y,z) to local (x,y,z);
    TMatrixD qQi(3,3);
    qQi(0,0) = cosRot;
    qQi(0,1) = sinRot;
    qQi(0,2) = 0.;
    qQi(1,0) = -sinRot;
    qQi(1,1) = cosRot;
    qQi(1,2) = 0.;
    qQi(2,0) = 0.;
    qQi(2,1) = 0.;
    qQi(2,2) = 1.;
   
    // covariance of fVert along the track  
    Double_t p[3],pt,ptot;
    t->GetPxPyPz(p);
    pt = TMath::Sqrt(p[0]*p[0]+p[1]*p[1]);
    ptot = TMath::Sqrt(pt*pt+p[2]*p[2]);
    Double_t cphi = p[0]/pt;               //cos(phi)=px/pt
    Double_t sphi = p[1]/pt;               //sin(phi)=py/pt
    Double_t clambda = pt/ptot;            //cos(lambda)=pt/ptot
    Double_t slambda = p[2]/ptot;            //sin(lambda)=pz/ptot
    Double_t covfVert[6];
    fVert.GetCovMatrix(covfVert);
    Double_t covfVertalongt = 
       covfVert[0]*cphi*cphi*clambda*clambda 
      +covfVert[1]*2.*cphi*sphi*clambda*clambda
      +covfVert[3]*2.*cphi*clambda*slambda 
      +covfVert[2]*sphi*sphi*clambda*clambda 
      +covfVert[4]*2.*sphi*clambda*slambda 
      +covfVert[5]*slambda*slambda; 
    // covariance matrix of local (x,y,z) - inverted
    TMatrixD uUi(3,3);
    uUi(0,0) = covfVertalongt * fnSigmaForUi00 * fnSigmaForUi00;
    AliDebug(1,Form("=====> sqrtUi00 cm  %f",TMath::Sqrt(uUi(0,0))));
    uUi(0,1) = 0.;
    uUi(0,2) = 0.;
    uUi(1,0) = 0.;
    uUi(1,1) = t->GetSigmaY2();
    uUi(1,2) = t->GetSigmaZY();
    uUi(2,0) = 0.;
    uUi(2,1) = t->GetSigmaZY();
    uUi(2,2) = t->GetSigmaZ2();
    //printf(" Ui :\n");
    //printf(" %f   %f\n",uUi(0,0),uUi(0,1));
    //printf(" %f   %f\n",uUi(1,0),uUi(1,1));
  
    if(uUi.Determinant() <= 0.) return kFALSE;
    TMatrixD uUiInv(TMatrixD::kInverted,uUi);
  
    // weights matrix: wWi = qQiT * uUiInv * qQi
    TMatrixD uUiInvQi(uUiInv,TMatrixD::kMult,qQi);
    TMatrixD m(qQi,TMatrixD::kTransposeMult,uUiInvQi);
    wWi = m;
  }


  return kTRUE;
} 
//---------------------------------------------------------------------------
void AliVertexerTracks::TooFewTracks() 
{
//
// When the number of tracks is < fMinTracks,
// deal with vertices not found and prepare to exit
//
  AliDebug(1,"TooFewTracks");

  Double_t pos[3],err[3];
  pos[0] = fNominalPos[0];
  err[0] = TMath::Sqrt(fNominalCov[0]);
  pos[1] = fNominalPos[1];
  err[1] = TMath::Sqrt(fNominalCov[2]);
  pos[2] = fNominalPos[2];
  err[2] = TMath::Sqrt(fNominalCov[5]);
  Int_t    ncontr = (err[0]>1. ? -1 : -3);
  if(fCurrentVertex) { delete fCurrentVertex; fCurrentVertex=0; }
  fCurrentVertex = new AliESDVertex(pos,err);
  fCurrentVertex->SetNContributors(ncontr);

  if(fConstraint) {
    fCurrentVertex->SetTitle("VertexerTracksWithConstraint");
  } else {
    fCurrentVertex->SetTitle("VertexerTracksNoConstraint");
  }

  if(!fTrkArraySel.IsEmpty()) fTrkArraySel.Delete(); 
  if(fIdSel) {delete [] fIdSel; fIdSel=NULL;}
  if(fTrksToSkip) {delete [] fTrksToSkip; fTrksToSkip=NULL;}

  return;
}
//---------------------------------------------------------------------------
void AliVertexerTracks::VertexFinder(Int_t optUseWeights) 
{

  // Get estimate of vertex position in (x,y) from tracks DCA
 
  Double_t initPos[3];
  initPos[2] = 0.;
  for(Int_t i=0;i<2;i++)initPos[i]=fNominalPos[i];
  Int_t nacc = (Int_t)fTrkArraySel.GetEntriesFast();
  Double_t aver[3]={0.,0.,0.};
  Double_t aversq[3]={0.,0.,0.};
  Double_t sigmasq[3]={0.,0.,0.};
  Double_t sigma=0;
  Int_t ncombi = 0;
  AliExternalTrackParam *track1;
  AliExternalTrackParam *track2;
  Double_t pos[3],dir[3]; 
  Double_t alpha,mindist;

  for(Int_t i=0; i<nacc; i++){
    track1 = (AliExternalTrackParam*)fTrkArraySel.At(i);
    alpha=track1->GetAlpha();
    mindist = TMath::Cos(alpha)*fNominalPos[0]+TMath::Sin(alpha)*fNominalPos[1];
    track1->GetXYZAt(mindist,GetFieldkG(),pos);
    track1->GetPxPyPzAt(mindist,GetFieldkG(),dir);
    AliStrLine *line1 = new AliStrLine(pos,dir); 

   //    AliStrLine *line1 = new AliStrLine();
   //    track1->ApproximateHelixWithLine(mindist,GetFieldkG(),line1);
   
    for(Int_t j=i+1; j<nacc; j++){
      track2 = (AliExternalTrackParam*)fTrkArraySel.At(j);
      alpha=track2->GetAlpha();
      mindist = TMath::Cos(alpha)*fNominalPos[0]+TMath::Sin(alpha)*fNominalPos[1];
      track2->GetXYZAt(mindist,GetFieldkG(),pos);
      track2->GetPxPyPzAt(mindist,GetFieldkG(),dir);
      AliStrLine *line2 = new AliStrLine(pos,dir); 
    //      AliStrLine *line2 = new AliStrLine();
    //  track2->ApproximateHelixWithLine(mindist,GetFieldkG(),line2);
      Double_t distCA=line2->GetDCA(line1);
      //printf("%d   %d   %f\n",i,j,distCA);
       if(fDCAcut<=0 || (fDCAcut>0&&distCA<fDCAcut)){
	Double_t pnt1[3],pnt2[3],crosspoint[3];

	if(optUseWeights<=0){
	  Int_t retcode = line2->Cross(line1,crosspoint);
	  if(retcode>=0){
	    ncombi++;
	    for(Int_t jj=0;jj<3;jj++)aver[jj]+=crosspoint[jj];
	    for(Int_t jj=0;jj<3;jj++)aversq[jj]+=(crosspoint[jj]*crosspoint[jj]);
	  }
	}
	if(optUseWeights>0){
	  Int_t retcode = line1->CrossPoints(line2,pnt1,pnt2);
	  if(retcode>=0){
	    Double_t cs, sn;
	    alpha=track1->GetAlpha();
	    cs=TMath::Cos(alpha); sn=TMath::Sin(alpha);	  
	    Double_t sx1=sn*sn*track1->GetSigmaY2(), sy1=cs*cs*track1->GetSigmaY2();
	    alpha=track2->GetAlpha();
	    cs=TMath::Cos(alpha); sn=TMath::Sin(alpha);
	    Double_t sx2=sn*sn*track2->GetSigmaY2(), sy2=cs*cs*track2->GetSigmaY2();
	    Double_t sz1=track1->GetSigmaZ2(), sz2=track2->GetSigmaZ2();
	    Double_t wx1=sx2/(sx1+sx2), wx2=1.- wx1;
	    Double_t wy1=sy2/(sy1+sy2), wy2=1.- wy1;
	    Double_t wz1=sz2/(sz1+sz2), wz2=1.- wz1;
	    crosspoint[0]=wx1*pnt1[0] + wx2*pnt2[0]; 
	    crosspoint[1]=wy1*pnt1[1] + wy2*pnt2[1]; 
	    crosspoint[2]=wz1*pnt1[2] + wz2*pnt2[2];
	  
	    ncombi++;
	    for(Int_t jj=0;jj<3;jj++)aver[jj]+=crosspoint[jj];
	    for(Int_t jj=0;jj<3;jj++)aversq[jj]+=(crosspoint[jj]*crosspoint[jj]);
	  }
	}
      }
      delete line2;
    }
    delete line1;
  }
  if(ncombi>0){
    for(Int_t jj=0;jj<3;jj++){
      initPos[jj] = aver[jj]/ncombi;
      //printf("%f\n",initPos[jj]);
      aversq[jj]/=ncombi;
      sigmasq[jj]=aversq[jj]-initPos[jj]*initPos[jj];
      sigma+=sigmasq[jj];
    }
    sigma=TMath::Sqrt(TMath::Abs(sigma));
  }
  else {
    Warning("VertexFinder","Finder did not succed");
    sigma=999;
  }
  fVert.SetXYZ(initPos);
  fVert.SetDispersion(sigma);
  fVert.SetNContributors(ncombi);
}
//---------------------------------------------------------------------------
void AliVertexerTracks::VertexFitter(Bool_t vfit, Bool_t chiCalc,Int_t useWeights) 
{
//
// The optimal estimate of the vertex position is given by a "weighted 
// average of tracks positions".
// Original method: V. Karimaki, CMS Note 97/0051
//
  const double kTiny = 1e-9;
  Bool_t useConstraint = fConstraint;
  Double_t initPos[3];
  if(!fOnlyFitter) {
    fVert.GetXYZ(initPos);
  } else {
    initPos[0]=fNominalPos[0];
    initPos[1]=fNominalPos[1];
    initPos[2]=fNominalPos[2];
  }

  Int_t nTrksSel = (Int_t)fTrkArraySel.GetEntries();
  if(nTrksSel==1) useConstraint=kTRUE;
  AliDebug(1,Form("--- VertexFitter(): start (%d,%d,%d)",vfit,chiCalc,useWeights));
  AliDebug(1,Form(" Number of tracks in array: %d\n",nTrksSel));
  AliDebug(1,Form(" Minimum # tracks required in fit: %d\n",fMinTracks));
  AliDebug(1,Form(" Vertex position after finder: %f,%f,%f\n",initPos[0],initPos[1],initPos[2]));
  if(useConstraint) AliDebug(1,Form(" This vertex will be used in fit: (%f+-%f,%f+-%f)\n",fNominalPos[0],TMath::Sqrt(fNominalCov[0]),fNominalPos[1],TMath::Sqrt(fNominalCov[2]))); 

  // special treatment for few-tracks fits (e.g. secondary vertices)
  Bool_t uUi3by3 = kFALSE; if(nTrksSel<5 && !useConstraint && !useWeights) uUi3by3 = kTRUE;

  Int_t i,j,k,step=0;
  static TMatrixD rv(3,1);
  static TMatrixD vV(3,3);
  rv(0,0) = initPos[0];
  rv(1,0) = initPos[1];
  rv(2,0) = initPos[2];
  Double_t xlStart,alpha;
  Int_t nTrksUsed = 0;
  Double_t chi2=0,chi2i,chi2b;
  AliExternalTrackParam *t = 0;
  Int_t failed = 0;

  // initial vertex covariance matrix
  TMatrixD vVb(3,3);
  vVb(0,0) = fNominalCov[0];
  vVb(0,1) = fNominalCov[1];
  vVb(0,2) = 0.;
  vVb(1,0) = fNominalCov[1];
  vVb(1,1) = fNominalCov[2];
  vVb(1,2) = 0.;
  vVb(2,0) = 0.;
  vVb(2,1) = 0.;
  vVb(2,2) = fNominalCov[5];
  TMatrixD vVbInv(TMatrixD::kInverted,vVb);
  TMatrixD rb(3,1);
  rb(0,0) = fNominalPos[0];
  rb(1,0) = fNominalPos[1];
  rb(2,0) = fNominalPos[2];
  TMatrixD vVbInvrb(vVbInv,TMatrixD::kMult,rb);
  //
  int currBC = fVert.GetBC();
  //
  // 2 steps:
  // 1st - estimate of vtx using all tracks
  // 2nd - estimate of global chi2
  //
  for(step=0; step<2; step++) {
    if      (step==0 && !vfit)    continue;
    else if (step==1 && !chiCalc) continue;
    chi2 = 0.;
    nTrksUsed = 0;
    fMVWSum = fMVWE2 = 0;
    if(step==1) { initPos[0]=rv(0,0); initPos[1]=rv(1,0); initPos[2]=rv(2,0);}
    AliDebug(2,Form("Step%d: inipos: %+f %+f %+f MinTr: %d, Sig2:%.2f)",step,initPos[0],initPos[1],initPos[2],fMinTracks,fMVSigma2));
    //
    TMatrixD sumWiri(3,1);
    TMatrixD sumWi(3,3);
    for(i=0; i<3; i++) {
      sumWiri(i,0) = 0.;
      for(j=0; j<3; j++) sumWi(j,i) = 0.;
    }

    // mean vertex constraint
    if(useConstraint) {
      for(i=0;i<3;i++) {
	sumWiri(i,0) += vVbInvrb(i,0);
	for(k=0;k<3;k++) sumWi(i,k) += vVbInv(i,k);
      }
      // chi2
      TMatrixD deltar = rv; deltar -= rb;
      TMatrixD vVbInvdeltar(vVbInv,TMatrixD::kMult,deltar);
      chi2b = deltar(0,0)*vVbInvdeltar(0,0)+
              deltar(1,0)*vVbInvdeltar(1,0)+
	      deltar(2,0)*vVbInvdeltar(2,0);
      chi2 += chi2b;
    }

    // loop on tracks  
    for(k=0; k<nTrksSel; k++) {
      //
      // get track from track array
      t = (AliExternalTrackParam*)fTrkArraySel.At(k);
      if (useWeights && t->TestBit(kBitUsed)) continue;
      //      
      int tBC = int(t->GetUniqueID()) - kTOFBCShift;    // BC assigned to this track
      if (fSelectOnTOFBunchCrossing) {
	if (!fKeepAlsoUnflaggedTOFBunchCrossing) continue;   // don't consider tracks with undefined BC	
	if (currBC!=AliVTrack::kTOFBCNA && tBC!=AliVTrack::kTOFBCNA && tBC!=currBC) continue;  // track does not match to current BCid
      }
      alpha = t->GetAlpha();
      xlStart = initPos[0]*TMath::Cos(alpha)+initPos[1]*TMath::Sin(alpha);
      // to vtxSeed (from finder)
      t->PropagateTo(xlStart,GetFieldkG());   
 
      // vector of track global coordinates
      TMatrixD ri(3,1);
      // covariance matrix of ri
      TMatrixD wWi(3,3);

      // get space point from track
      if(!TrackToPoint(t,ri,wWi,uUi3by3)) continue;

      // track chi2
      TMatrixD deltar = rv; deltar -= ri;
      TMatrixD wWideltar(wWi,TMatrixD::kMult,deltar);
      chi2i = deltar(0,0)*wWideltar(0,0)+
              deltar(1,0)*wWideltar(1,0)+
	      deltar(2,0)*wWideltar(2,0);
      //
      if (useWeights) {
	//double sg = TMath::Sqrt(fMVSigma2);
	//double chi2iw = (deltar(0,0)*wWideltar(0,0)+deltar(1,0)*wWideltar(1,0))/sg + deltar(2,0)*wWideltar(2,0)/fMVSigma2;
	//double wgh = (1-chi2iw/fMVTukey2); 
	double chi2iw = chi2i;
	double wgh = (1-chi2iw/fMVTukey2/fMVSigma2); 

	if (wgh<kTiny) wgh = 0;
	else if (useWeights==2) wgh = 1.; // use as binary weight
	if (step==1) ((AliESDtrack*)t)->SetBit(kBitUsed, wgh>0);
	if (wgh<kTiny) continue; // discard the track
	wWi *= wgh;  // RS: use weight?
	fMVWSum += wgh;
	fMVWE2  += wgh*chi2iw;
      }
      // add to total chi2
      if (fSelectOnTOFBunchCrossing && tBC!=AliVTrack::kTOFBCNA && currBC<0) currBC = tBC;
      //
      chi2 += chi2i;
      TMatrixD wWiri(wWi,TMatrixD::kMult,ri); 
      sumWiri += wWiri;
      sumWi   += wWi;

      nTrksUsed++;
    } // end loop on tracks

    if(nTrksUsed < fMinTracks) {
      failed=1;
      continue;
    }

    Double_t determinant = sumWi.Determinant();
    if(determinant < fMinDetFitter)  { 
      AliDebug(1,Form("det(V) = %f (<%f)\n",determinant,fMinDetFitter));       
      failed=1;
      continue;
    }

    if(step==0) { 
      // inverted of weights matrix
      TMatrixD invsumWi(TMatrixD::kInverted,sumWi);
      vV = invsumWi;
      // position of primary vertex
      rv.Mult(vV,sumWiri);
    }
  } // end loop on the 2 steps

  if(failed) { 
    fVert.SetNContributors(-1);
    if (chiCalc) {
      TooFewTracks();
      if (fCurrentVertex) fVert = *fCurrentVertex;  // RS
    }
    return; 
  }
  //
  Double_t position[3];
  position[0] = rv(0,0);
  position[1] = rv(1,0);
  position[2] = rv(2,0);
  Double_t covmatrix[6];
  covmatrix[0] = vV(0,0);
  covmatrix[1] = vV(0,1);
  covmatrix[2] = vV(1,1);
  covmatrix[3] = vV(0,2);
  covmatrix[4] = vV(1,2);
  covmatrix[5] = vV(2,2);
  
  // for correct chi2/ndf, count constraint as additional "track"
  if(fConstraint) nTrksUsed++;
  //
  if (vfit && !chiCalc) { // RS: special mode for multi-vertex finder
    fVert.SetXYZ(position);
    fVert.SetCovarianceMatrix(covmatrix);
    fVert.SetNContributors(nTrksUsed);
    return;
  } 
  // store data in the vertex object
  if(fCurrentVertex) { delete fCurrentVertex; fCurrentVertex=0; }
  fCurrentVertex = new AliESDVertex(position,covmatrix,chi2,nTrksUsed);
  fCurrentVertex->SetBC(currBC);
  fVert = *fCurrentVertex;  // RS
  AliDebug(1," Vertex after fit:");
  AliDebug(1,Form("xyz: %f %f %f; nc %d",fCurrentVertex->GetX(),fCurrentVertex->GetY(),fCurrentVertex->GetZ(),fCurrentVertex->GetNContributors()));
  AliDebug(1,"--- VertexFitter(): finish\n");
 

  return;
}
//----------------------------------------------------------------------------
AliESDVertex* AliVertexerTracks::VertexForSelectedTracks(const TObjArray *trkArray,
							 UShort_t *id,
							 Bool_t optUseFitter,
							 Bool_t optPropagate,
							 Bool_t optUseDiamondConstraint) 
{
//
// Return vertex from tracks (AliExternalTrackParam) in array
//
  fCurrentVertex = 0;
  // set optUseDiamondConstraint=TRUE only if you are reconstructing the 
  // primary vertex!
  if(optUseDiamondConstraint) {
    SetConstraintOn();
  } else {    
    SetConstraintOff();
  }

  // get tracks and propagate them to initial vertex position
  fIdSel = new UShort_t[(Int_t)trkArray->GetEntriesFast()];
  Int_t nTrksSel = PrepareTracks(*trkArray,id,0);
  if((!optUseDiamondConstraint && nTrksSel<TMath::Max(2,fMinTracks)) ||
     (optUseDiamondConstraint && nTrksSel<1)) {
    TooFewTracks();
    return fCurrentVertex;
  }
 
  // vertex finder
  if(nTrksSel==1) {
    AliDebug(1,"Just one track");
    OneTrackVertFinder();
  } else {
    switch (fAlgo) {
      case 1: StrLinVertexFinderMinDist(1); break;
      case 2: StrLinVertexFinderMinDist(0); break;
      case 3: HelixVertexFinder();          break;
      case 4: VertexFinder(1);              break;
      case 5: VertexFinder(0);              break;
      default: printf("Wrong algorithm\n"); break;  
    }
  }
  AliDebug(1," Vertex finding completed\n");
  Double_t vdispersion=fVert.GetDispersion();

  // vertex fitter
  if(optUseFitter) {
    VertexFitter();
  } else {
    Double_t position[3]={fVert.GetX(),fVert.GetY(),fVert.GetZ()};
    Double_t covmatrix[6];
    fVert.GetCovMatrix(covmatrix);
    Double_t chi2=99999.;
    Int_t    nTrksUsed=fVert.GetNContributors();
    fCurrentVertex = new AliESDVertex(position,covmatrix,chi2,nTrksUsed);    
  }
  fCurrentVertex->SetDispersion(vdispersion);


  // set indices of used tracks and propagate track to found vertex
  UShort_t *indices = 0;
  Double_t d0z0[2],covd0z0[3];
  AliExternalTrackParam *t = 0;
  if(fCurrentVertex->GetNContributors()>0) {
    indices = new UShort_t[fTrkArraySel.GetEntriesFast()];
    for(Int_t jj=0; jj<(Int_t)fTrkArraySel.GetEntriesFast(); jj++) {
      indices[jj] = fIdSel[jj];
      t = (AliExternalTrackParam*)fTrkArraySel.At(jj);
      if(optPropagate && optUseFitter) {
	if(TMath::Sqrt(fCurrentVertex->GetX()*fCurrentVertex->GetX()+fCurrentVertex->GetY()*fCurrentVertex->GetY())<3.) {
	  t->PropagateToDCA(fCurrentVertex,GetFieldkG(),100.,d0z0,covd0z0);
	  AliDebug(1,Form("Track %d propagated to found vertex",jj));
	} else {
	  AliWarning("Found vertex outside beam pipe!");
	}
      }
    }
    fCurrentVertex->SetIndices(fCurrentVertex->GetNContributors(),indices);
  }

  // clean up
  if (indices) {delete [] indices; indices=NULL;}
  delete [] fIdSel; fIdSel=NULL;
  fTrkArraySel.Delete();
  
  return fCurrentVertex;
}
 
//----------------------------------------------------------------------------
AliESDVertex* AliVertexerTracks::VertexForSelectedESDTracks(TObjArray *trkArray,
							     Bool_t optUseFitter,
							    Bool_t optPropagate,
							    Bool_t optUseDiamondConstraint)

{
//
// Return vertex from array of ESD tracks
//

  Int_t nTrks = (Int_t)trkArray->GetEntriesFast();
  UShort_t *id = new UShort_t[nTrks];

  AliESDtrack *esdt = 0;
  for(Int_t i=0; i<nTrks; i++){
    esdt = (AliESDtrack*)trkArray->At(i);
    id[i] = (UShort_t)esdt->GetID();
  }
    
  VertexForSelectedTracks(trkArray,id,optUseFitter,optPropagate,optUseDiamondConstraint);

  delete [] id; id=NULL;

  return fCurrentVertex;
}
 
//______________________________________________________
Bool_t AliVertexerTracks::FindNextVertexMV()
{
  // try to find a new vertex
  fMVSigma2 = fMVSig2Ini;
  double prevSig2 = fMVSigma2;
  double minDst2 = fMVMinDst*fMVMinDst;
  const double kSigLimit = 1.0;
  const double kSigLimitE = kSigLimit+1e-6;
  const double kPushFactor = 0.5;
  const int kMaxIter = 20;
  double push = kPushFactor;
  //
  int iter = 0;
  double posP[3]={0,0,0},pos[3]={0,0,0};
  fVert.GetXYZ(posP);
  //
  
  do {    
    fVert.SetBC(AliVTrack::kTOFBCNA);
    VertexFitter(kTRUE,kFALSE,1);
    if (fVert.GetNContributors()<fMinTracks) {
      AliDebug(3,Form("Failed in iteration %d: No Contributirs",iter)); 
      break;
    } // failed
    if (fMVWSum>0) fMVSigma2 = TMath::Max(fMVWE2/fMVWSum,kSigLimit);
    else {
      AliDebug(3,Form("Failed %d, no weithgs",iter)); 
      iter = kMaxIter+1; 
      break;
    } // failed
    //
    double sigRed = (prevSig2-fMVSigma2)/prevSig2; // reduction of sigma2
    //
    fVert.GetXYZ(pos);
    double dst2 = (pos[0]-posP[0])*(pos[0]-posP[0])+(pos[1]-posP[1])*(pos[1]-posP[1])+(pos[2]-posP[2])*(pos[2]-posP[2]);
    AliDebug(3,Form("It#%2d Vtx: %+f %+f %+f Dst:%f Sig: %f [%.2f/%.2f] SigRed:%f",iter,pos[0],pos[1],pos[2],TMath::Sqrt(dst2),fMVSigma2,fMVWE2,fMVWSum,sigRed));
    if ( (++iter<kMaxIter) && (sigRed<0 || sigRed<fMVMinSig2Red) && fMVSigma2>fMVMaxSigma2) {
      fMVSigma2 *= push; // stuck, push little bit
      push *= kPushFactor;
      if (fMVSigma2<1.) fMVSigma2 = 1.; 
      AliDebug(3,Form("Pushed sigma2 to %f",fMVSigma2));
    }
    else if (dst2<minDst2 && ((sigRed<0 || sigRed<fMVMinSig2Red) || fMVSigma2<kSigLimitE)) break;
    //
    fVert.GetXYZ(posP); // fetch previous vertex position
    prevSig2 = fMVSigma2;
  } while(iter<kMaxIter);
  //
  if (fVert.GetNContributors()<0 || iter>kMaxIter || fMVSigma2>fMVMaxSigma2) {
    return kFALSE;
  }
  else {
    VertexFitter(kFALSE,kTRUE,fMVFinalWBinary ? 2:1); // final chi2 calculation
    int nv = fMVVertices->GetEntries();
    // create indices
    int ntrk = fTrkArraySel.GetEntries();
    int nindices = fCurrentVertex->GetNContributors() - (fConstraint ? 1:0);
    if (nindices<1) {
      delete fCurrentVertex;
      fCurrentVertex = 0;
      return kFALSE;
    }
    UShort_t *indices = 0;
    if (nindices>0) indices = new UShort_t[nindices];
    int nadded = 0;
    for (int itr=0;itr<ntrk;itr++) {
      AliExternalTrackParam* t = (AliExternalTrackParam*)fTrkArraySel[itr];
      if (t->TestBit(kBitAccounted) || !t->TestBit(kBitUsed)) continue;   // already belongs to some vertex
      t->SetBit(kBitAccounted);
      indices[nadded++] = fIdSel[itr];
    }
    if (nadded!=nindices) {
      printf("Mismatch : NInd: %d Nadd: %d\n",nindices,nadded);
    }
    fCurrentVertex->SetIndices(nadded,indices);
    // set vertex title
    TString title="VertexerTracksMVNoConstraint";
    if(fConstraint) title="VertexerTracksMVWithConstraint";
    fCurrentVertex->SetTitle(title.Data());    
    fMVVertices->AddLast(fCurrentVertex);
    AliDebug(3,Form("Added new vertex #%d NCont:%d XYZ: %f %f %f",nindices,nv,fCurrentVertex->GetX(),fCurrentVertex->GetY(),fCurrentVertex->GetZ()));
    if (indices) delete[] indices;
    fCurrentVertex = 0; // already attached to fMVVertices
    return kTRUE;
  }
}

//______________________________________________________
void AliVertexerTracks::FindVerticesMV()
{
  // find and fit multiple vertices
  // 
  double step = fMVScanStep>1 ?  fMVScanStep : 1.;
  double zmx = 3*TMath::Sqrt(fNominalCov[5]);
  double zmn = -zmx;
  int nz = TMath::Nint((zmx-zmn)/step); if (nz<1) nz=1;
  double dz = (zmx-zmn)/nz;
  int izStart=0;
  AliDebug(2,Form("%d seeds between %f and %f",nz,zmn+dz/2,zmx+dz/2));
  //
  if (!fMVVertices) fMVVertices = new TObjArray(10);
  fMVVertices->Clear();
  //
  int ntrLeft = (Int_t)fTrkArraySel.GetEntries();
  //
  double sig2Scan = fMVSig2Ini;
  Bool_t runMore = kTRUE;
  int cntWide = 0;
  while (runMore) {
    fMVSig2Ini = sig2Scan*1e3;  // try wide search
    Bool_t found = kFALSE;
    cntWide++;
    fVert.SetNContributors(-1);
    fVert.SetXYZ(fNominalPos);
    AliDebug(3,Form("Wide search #%d Z= %f Sigma2=%f",cntWide,fNominalPos[2],fMVSig2Ini));
    if (FindNextVertexMV()) { // are there tracks left to consider?
      AliESDVertex* vtLast = (AliESDVertex*)fMVVertices->Last();
      if (vtLast && vtLast->GetNContributors()>0) ntrLeft -= vtLast->GetNContributors()-(fConstraint ? 1:0);
      if (ntrLeft<1) runMore = kFALSE; 
      found = kTRUE;
      continue;
    }  
    // if nothing is found, do narrow sig2ini scan
    fMVSig2Ini = sig2Scan;
    for (;izStart<nz;izStart++) {
      double zSeed = zmn+dz*(izStart+0.5);
      AliDebug(3,Form("Seed %d: Z= %f Sigma2=%f",izStart,zSeed,fMVSig2Ini));
      fVert.SetNContributors(-1);
      fVert.SetXYZ(fNominalPos);
      fVert.SetZv(zSeed);
      if (FindNextVertexMV()) { // are there tracks left to consider?
	AliESDVertex* vtLast = (AliESDVertex*)fMVVertices->Last();
	if (vtLast && vtLast->GetNContributors()>0) ntrLeft -= vtLast->GetNContributors()-(fConstraint ? 1:0);
	if (ntrLeft<1) runMore = kFALSE;
	found = kTRUE;
	break;
      }    
    }
    runMore = found; // if nothing was found, no need for new iteration
  }
  fMVSig2Ini = sig2Scan;
  int nvFound = fMVVertices->GetEntriesFast();
  AliDebug(2,Form("Number of found vertices: %d",nvFound));
  if (nvFound<1) TooFewTracks();
  if (AliLog::GetGlobalDebugLevel()>0) fMVVertices->Print();
  //
}

//______________________________________________________
void AliVertexerTracks::AnalyzePileUp(AliESDEvent* esdEv)
{
  // if multiple vertices are found, try to find the primary one and attach it as fCurrentVertex
  // then attach pile-up vertices directly to esdEv
  //
  int nFND = (fMVVertices && fMVVertices->GetEntriesFast()) ? fMVVertices->GetEntriesFast() : 0;
  if (nFND<1) { if (!fCurrentVertex) TooFewTracks(); return;} // no multiple vertices
  //
  int indCont[nFND];
  int nIndx[nFND];
  for (int iv=0;iv<nFND;iv++) {
    AliESDVertex* fnd = (AliESDVertex*)fMVVertices->At(iv);
    indCont[iv] = iv;
    nIndx[iv]   = fnd->GetNIndices();
  }
  TMath::Sort(nFND, nIndx, indCont, kTRUE); // sort in decreasing order of Nindices
  double dists[nFND];
  int    distOrd[nFND],indx[nFND];
  for (int iv=0;iv<nFND;iv++) {
    AliESDVertex* fndI = (AliESDVertex*)fMVVertices->At(indCont[iv]);
    if (fndI->GetStatus()<1) continue; // discarded
    int ncomp = 0;
    for (int jv=iv+1;jv<nFND;jv++) {
      AliESDVertex* fndJ = (AliESDVertex*)fMVVertices->At(indCont[jv]);
      if (fndJ->GetStatus()<1) continue;
      dists[ncomp] = fndI->GetWDist(fndJ)*fndJ->GetNIndices();
      distOrd[ncomp] = indCont[jv];
      indx[ncomp]  = ncomp;
      ncomp++;
    }
    if (ncomp<1) continue;
    TMath::Sort(ncomp, dists, indx, kFALSE); // sort in increasing distance order
    for (int jv0=0;jv0<ncomp;jv0++) {
      int jv = distOrd[indx[jv0]];
      AliESDVertex* fndJ = (AliESDVertex*)fMVVertices->At(jv);
      if (dists[indx[jv0]]<fMVMaxWghNtr) { // candidate for split vertex
	//before eliminating the small close vertex, check if we should transfere its BC to largest one
	if (fndJ->GetBC()!=AliVTrack::kTOFBCNA && fndI->GetBC()==AliVTrack::kTOFBCNA) fndI->SetBC(fndJ->GetBC());
	//
	// leave the vertex only if both vertices have definit but different BC's, then this is not splitting.
	if ( (fndJ->GetBC()==fndI->GetBC()) || (fndJ->GetBC()==AliVTrack::kTOFBCNA)) fndJ->SetNContributors(-fndJ->GetNContributors());
      }
    }
  }
  //
  // select as a primary the largest vertex with BC=0, or the largest with BC non-ID
  int primBC0=-1,primNoBC=-1;
  for (int iv0=0;iv0<nFND;iv0++) {
    int iv = indCont[iv0];
    AliESDVertex* fndI = (AliESDVertex*)fMVVertices->At(iv);
    if (!fndI) continue;
    if (fndI->GetStatus()<1) {fMVVertices->RemoveAt(iv); delete fndI; continue;}
    if (primBC0<0  && fndI->GetBC()==0) primBC0 = iv;
    if (primNoBC<0 && fndI->GetBC()==AliVTrack::kTOFBCNA)  primNoBC = iv;
  }
  //
  if (primBC0>=0) fCurrentVertex = (AliESDVertex*)fMVVertices->At(primBC0);
  if (!fCurrentVertex && primNoBC>=0) fCurrentVertex = (AliESDVertex*)fMVVertices->At(primNoBC);
  if (fCurrentVertex) fMVVertices->Remove(fCurrentVertex);
  else {  // all vertices have BC>0, no primary vertex
    fCurrentVertex = new AliESDVertex();
    fCurrentVertex->SetNContributors(-3);
    fCurrentVertex->SetBC(AliVTrack::kTOFBCNA);
  }
  fCurrentVertex->SetID(-1);
  //
  // add pileup vertices
  int nadd = 0;
  for (int iv0=0;iv0<nFND;iv0++) {
    int iv = indCont[iv0];
    AliESDVertex* fndI = (AliESDVertex*)fMVVertices->At(iv);
    if (!fndI) continue;
    fndI->SetID(++nadd);
    esdEv->AddPileupVertexTracks(fndI);
  }
  //
  fMVVertices->Delete();
  //
}

//______________________________________________________
void AliVertexerTracks::FindAllVertices(Int_t nTrksOrig, 
					const TObjArray *trkArrayOrig,
					Double_t* zTr, 
					Double_t* err2zTr, 
					UShort_t* idOrig){

  // clusterize tracks using z coordinates of intersection with beam axis
  // and compute all vertices 

  UShort_t* posOrig=new UShort_t[nTrksOrig];
  for(Int_t iTr=0; iTr<nTrksOrig; iTr++) posOrig[iTr]=iTr;
 

  // sort points along Z
  AliDebug(1,Form("Sort points along Z, used tracks %d",nTrksOrig));
  for(Int_t iTr1=0; iTr1<nTrksOrig; iTr1++){
    for(Int_t iTr2=iTr1+1; iTr2<nTrksOrig; iTr2++){
      if(zTr[iTr1]>zTr[iTr2]){
	Double_t tmpz=zTr[iTr2];
	Double_t tmperr=err2zTr[iTr2];
	UShort_t tmppos=posOrig[iTr2];
	UShort_t tmpid=idOrig[iTr2];
	zTr[iTr2]=zTr[iTr1];
	err2zTr[iTr2]=err2zTr[iTr1];
	posOrig[iTr2]=posOrig[iTr1];
	idOrig[iTr2]=idOrig[iTr1];
	zTr[iTr1]=tmpz;
	err2zTr[iTr1]=tmperr;
	idOrig[iTr1]=tmpid;
	posOrig[iTr1]=tmppos;
      }
    }
  }

  // clusterize
  Int_t nClusters=0;
  Int_t* firstTr=new Int_t[nTrksOrig];
  Int_t* lastTr=new Int_t[nTrksOrig];

  firstTr[0]=0;
  for(Int_t iTr=0; iTr<nTrksOrig-1; iTr++){
    Double_t distz=zTr[iTr+1]-zTr[iTr];
    Double_t errdistz=TMath::Sqrt(err2zTr[iTr+1]+err2zTr[iTr]);
    if(errdistz<=0.000001) errdistz=0.000001;
    if(distz>fDeltaZCutForCluster || (distz/errdistz)>fnSigmaZCutForCluster){
      lastTr[nClusters]=iTr;
      firstTr[nClusters+1]=iTr+1;
      nClusters++;
    }
  }
  lastTr[nClusters]=nTrksOrig-1;

  // Compute vertices
  AliDebug(1,Form("Number of found clusters %d",nClusters+1));
  Int_t nFoundVertices=0;

  if (!fMVVertices) fMVVertices = new TObjArray(nClusters+1);

  fMVVertices->Clear();
  TObjArray cluTrackArr(nTrksOrig);
  UShort_t *idTrkClu=new UShort_t[nTrksOrig];
  //  Int_t maxContr=0;
  //  Int_t maxPos=-1;

  for(Int_t iClu=0; iClu<=nClusters; iClu++){
    Int_t nCluTracks=lastTr[iClu]-firstTr[iClu]+1;
    cluTrackArr.Clear();
    AliDebug(1,Form(" Vertex #%d tracks %d first tr %d  last track %d",iClu,nCluTracks,firstTr[iClu],lastTr[iClu]));
    Int_t nSelTr=0;
    for(Int_t iTr=firstTr[iClu]; iTr<=lastTr[iClu]; iTr++){
      AliExternalTrackParam* t=(AliExternalTrackParam*)trkArrayOrig->At(posOrig[iTr]);
      if(TMath::Abs(t->GetZ()-zTr[iTr])>0.00001){
	AliError(Form("Clu %d Track %d zTrack=%f  zVec=%f\n",iClu,iTr,t->GetZ(),zTr[iTr]));
      }
      cluTrackArr.AddAt(t,nSelTr);
      idTrkClu[nSelTr]=idOrig[iTr];
      AliDebug(1,Form("   Add track %d: id %d, z=%f",iTr,idOrig[iTr],zTr[iTr]));
      nSelTr++;
    }
    AliESDVertex* vert=FindPrimaryVertex(&cluTrackArr,idTrkClu);
    AliDebug(1,Form("Found vertex in z=%f with %d contributors",vert->GetZ(),
		 vert->GetNContributors()));

    fCurrentVertex=0;
    if(vert->GetNContributors()>0){
      nFoundVertices++;
      fMVVertices->AddLast(vert);
    }
    //    if(vert->GetNContributors()>maxContr){
    //      maxContr=vert->GetNContributors();
    //      maxPos=nFoundVertices-1;
    //    }
  }

  AliDebug(1,Form("Number of found vertices %d (%d)",nFoundVertices,fMVVertices->GetEntriesFast()));
  // if(maxPos>=0 && maxContr>0){
  //   AliESDVertex* vtxMax=(AliESDVertex*)fMVVertices->At(maxPos);
  //   if(fCurrentVertex) delete fCurrentVertex; 
  //   fCurrentVertex=new AliESDVertex(*vtxMax);
  // }

  delete [] firstTr;
  delete [] lastTr;
  delete [] idTrkClu;
  delete [] posOrig;

  return;

}
