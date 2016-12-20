/// \file CookedTracker.cxx
/// \brief Implementation of the "Cooked Matrix" ITS tracker
/// \author iouri.belikov@cern.ch

//-------------------------------------------------------------------------
//                     A stand-alone ITS tracker
//    The pattern recongintion based on the "cooked covariance" approach
//-------------------------------------------------------------------------

#include <vector>

#include <TTree.h>
#include <TClonesArray.h>

#include "AliLog.h"
#include "ITSReconstruction/Cluster.h"
#include "ITSReconstruction/CookedTracker.h"
#include "ITSReconstruction/CookedTrack.h" 

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace AliceO2::ITS;

//************************************************
// Constants hardcoded for the moment:
//************************************************
// seed "windows" in z and phi: MakeSeeds
const Double_t kzWin=0.33;
const Double_t kminPt=0.05;
// Maximal accepted impact parameters for the seeds 
const Double_t kmaxDCAxy=3.;
const Double_t kmaxDCAz= 3.;
// Layers for the seeding
const Int_t kSeedingLayer1=6, kSeedingLayer2=4, kSeedingLayer3=5;
// Space point resolution
const Double_t kSigma2=0.0005*0.0005;
// Max accepted chi2
const Double_t kmaxChi2PerCluster=20.;
const Double_t kmaxChi2PerTrack=30.;
// Tracking "road" from layer to layer
const Double_t kRoadY=0.2;
const Double_t kRoadZ=0.7;
// Minimal number of attached clusters
const Int_t kminNumberOfClusters=4;

//************************************************
// TODO:
//************************************************
// Seeding:
// Precalculate cylidnrical (r,phi) for the clusters;
// use exact r's for the clusters


CookedTracker::AliITSUlayer
              CookedTracker::fgLayers[CookedTracker::kNLayers];

CookedTracker::CookedTracker(): 
fSeeds(0),
fSAonly(kTRUE) 
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
  const Double_t 
  klRadius[7]={2.34, 3.15, 3.93, 19.61, 24.55, 34.39, 39.34}; //tdr6

  for (Int_t i=0; i<kNLayers; i++) fgLayers[i].SetR(klRadius[i]);

  // Some default primary vertex
  Double_t xyz[]={0.,0.,0.};
  Double_t ers[]={2.,2.,2.};

  SetVertex(xyz,ers);

}

CookedTracker::~CookedTracker() 
{
  //--------------------------------------------------------------------
  // Virtual destructor
  //--------------------------------------------------------------------

  if (fSeeds) fSeeds->Delete(); delete fSeeds; 

}

static Double_t 
f1(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t x3, Double_t y3)
{
    //-----------------------------------------------------------------
    // Initial approximation of the track curvature
    //-----------------------------------------------------------------
    Double_t d=(x2-x1)*(y3-y2)-(x3-x2)*(y2-y1);
    Double_t a=0.5*((y3-y2)*(y2*y2-y1*y1+x2*x2-x1*x1)-
                    (y2-y1)*(y3*y3-y2*y2+x3*x3-x2*x2));
    Double_t b=0.5*((x2-x1)*(y3*y3-y2*y2+x3*x3-x2*x2)-
                    (x3-x2)*(y2*y2-y1*y1+x2*x2-x1*x1));
    
    Double_t xr=TMath::Abs(d/(d*x1-a)), yr=TMath::Abs(d/(d*y1-b));
    
    Double_t crv=xr*yr/sqrt(xr*xr+yr*yr);
    if (d>0) crv=-crv;

    return crv;
}

static Double_t 
f2(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t x3, Double_t y3)
{
    //-----------------------------------------------------------------
    // Initial approximation of the x-coordinate of the center of curvature 
    //-----------------------------------------------------------------

  Double_t k1=(y2-y1)/(x2-x1), k2=(y3-y2)/(x3-x2);
  Double_t x0=0.5*(k1*k2*(y1-y3) + k2*(x1+x2) - k1*(x2+x3))/(k2-k1);

  return x0;
}

static Double_t 
f3(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t z1, Double_t z2)
{
    //-----------------------------------------------------------------
    // Initial approximation of the tangent of the track dip angle
    //-----------------------------------------------------------------
    return (z1 - z2)/sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

Bool_t CookedTracker::
AddCookedSeed(const Float_t r1[3], Int_t l1, Int_t i1, 
              const Float_t r2[3], Int_t l2, Int_t i2,
              const Cluster *c3,Int_t l3, Int_t i3) 
{
    //--------------------------------------------------------------------
    // This is the main cooking function.
    // Creates seed parameters out of provided clusters.
    //--------------------------------------------------------------------
    Float_t x,a;
    if (!c3->GetXAlphaRefPlane(x,a)) return kFALSE;

    Double_t ca=TMath::Cos(a), sa=TMath::Sin(a);
    Double_t x1 = r1[0]*ca + r1[1]*sa,
             y1 =-r1[0]*sa + r1[1]*ca, z1 = r1[2];
    Double_t x2 = r2[0]*ca + r2[1]*sa,
             y2 =-r2[0]*sa + r2[1]*ca, z2 = r2[2];
    Double_t x3 = x,  y3 = c3->GetY(), z3 = c3->GetZ();

    Double_t par[5];
    par[0]=y3;
    par[1]=z3;
    Double_t crv=f1(x1, y1, x2, y2, x3, y3); //curvature
    Double_t x0 =f2(x1, y1, x2, y2, x3, y3); //x-coordinate of the center
    Double_t tgl12=f3(x1, y1, x2, y2, z1, z2);
    Double_t tgl23=f3(x2, y2, x3, y3, z2, z3);

    Double_t sf=crv*(x-x0);
    if (TMath::Abs(sf) >= kAlmost1) return kFALSE;
    par[2]=sf;

    par[3]=0.5*(tgl12 + tgl23);
    Double_t bz=GetBz();
    par[4]=(TMath::Abs(bz) < kAlmost0Field) ? kAlmost0 : crv/(bz*kB2C);

    Double_t cov[15];
    /*
    for (Int_t i=0; i<15; i++) cov[i]=0.;
    cov[0] =kSigma2*10;
    cov[2] =kSigma2*10;
    cov[5] =0.007*0.007*10;   //FIXME all these lines
    cov[9] =0.007*0.007*10;
    cov[14]=0.1*0.1*10;
    */
    const Double_t dlt=0.0005;
    Double_t
    fy=1./(fgLayers[kSeedingLayer3].GetR() - fgLayers[kSeedingLayer2].GetR());
    Double_t tz=fy;
    Double_t cy=(f1(x1, y1, x2, y2+dlt, x3, y3) - crv)/dlt/bz/kB2C;
    cy*=20; //FIXME: MS contribution to the cov[14]
    Double_t s2=kSigma2;

    cov[0]=s2;
    cov[1]=0.;     cov[2]=s2;
    cov[3]=s2*fy;  cov[4]=0.;    cov[5]=s2*fy*fy;
    cov[6]=0.;     cov[7]=s2*tz; cov[8]=0.;        cov[9]=s2*tz*tz;
    cov[10]=s2*cy; cov[11]=0.;   cov[12]=s2*fy*cy; cov[13]=0.; cov[14]=s2*cy*cy;

    CookedTrack *seed=new CookedTrack();
    seed->Set(Double_t(x), Double_t(a), par, cov);

    Float_t dz[2]; 
    seed->GetDZ(GetX(),GetY(),GetZ(),GetBz(),dz);
    if (TMath::Abs(dz[0]) > kmaxDCAxy) {delete seed; return kFALSE;} 
    if (TMath::Abs(dz[1]) > kmaxDCAz ) {delete seed; return kFALSE;} 

    Double_t xx0 = 0.008; // Rough layer thickness
    Double_t radl= 9.36;  // Radiation length of Si [cm]
    Double_t rho = 2.33;  // Density of Si [g/cm^3] 
    Double_t mass= 0.139; // Pion
    if (!seed->CorrectForMeanMaterial(xx0, xx0*radl*rho, mass, kTRUE)) {
       delete seed; return kFALSE;
    }

    seed->SetClusterIndex(l1,i1);
    seed->SetClusterIndex(l2,i2);
    seed->SetClusterIndex(l3,i3);

#pragma omp critical
    fSeeds->AddLast(seed);

    return kTRUE;
}

Int_t CookedTracker::MakeSeeds() {
  //--------------------------------------------------------------------
  // This is the main pattern recongition function.
  // Creates seeds out of two clusters and another point.
  //--------------------------------------------------------------------
   const Double_t zv=GetZ();

   AliITSUlayer &layer1=fgLayers[kSeedingLayer1];
   AliITSUlayer &layer2=fgLayers[kSeedingLayer2];
   AliITSUlayer &layer3=fgLayers[kSeedingLayer3];

   const Double_t maxC  = TMath::Abs(GetBz()*kB2C/kminPt);
   const Double_t kpWin = TMath::ASin(0.5*maxC*layer1.GetR()) - 
                          TMath::ASin(0.5*maxC*layer2.GetR());

   Int_t nClusters1=layer1.GetNumberOfClusters();
   Int_t nClusters2=layer2.GetNumberOfClusters();
   Int_t nClusters3=layer3.GetNumberOfClusters();
#pragma omp parallel for
   for (Int_t n1=0; n1<nClusters1; n1++) {
     Cluster *c1=layer1.GetCluster(n1);
     //
     //Int_t lab=c1->GetLabel(0);
     //
     Double_t z1=c1->GetZ();
     Float_t xyz1[3]; c1->GetGlobalXYZ(xyz1);
     Double_t r1=TMath::Sqrt(xyz1[0]*xyz1[0] + xyz1[1]*xyz1[1]);
     Double_t phi1=layer1.GetClusterPhi(n1);

     Double_t zr2=zv + layer2.GetR()/r1*(z1-zv);
     Int_t start2=layer2.FindClusterIndex(zr2-kzWin);
     for (Int_t n2=start2; n2<nClusters2; n2++) {
         Cluster *c2=layer2.GetCluster(n2);
         //
         //if (c2->GetLabel(0)!=lab) continue;
	 //
         Double_t z2=c2->GetZ();
         if (z2 > (zr2+kzWin)) break;  //check in Z

         Double_t phi2=layer2.GetClusterPhi(n2);
         if (TMath::Abs(phi2-phi1) > kpWin) continue;  //check in Phi

         Float_t xyz2[3]; c2->GetGlobalXYZ(xyz2);
         Double_t r2=TMath::Sqrt(xyz2[0]*xyz2[0] + xyz2[1]*xyz2[1]);
         Double_t crv=f1(xyz1[0], xyz1[1], xyz2[0], xyz2[1], GetX(),GetY());

         Double_t zr3=z1 + (layer3.GetR()-r1)/(r2-r1)*(z2-z1);
	 Double_t dz=kzWin/2;
         Int_t start3=layer3.FindClusterIndex(zr3-dz);
         for (Int_t n3=start3; n3<nClusters3; n3++) {
             Cluster *c3=layer3.GetCluster(n3);
             //
             //if (c3->GetLabel(0)!=lab) continue;
             //
             Double_t z3=c3->GetZ();
             if (z3 > (zr3+dz)) break;  //check in Z

             Double_t r3=layer3.GetXRef(n3);
             Double_t phir3 = phi1 + 0.5*crv*(r3 - r1); 
             Double_t phi3=layer3.GetClusterPhi(n3);
             if (TMath::Abs(phir3-phi3) > kpWin/100) continue;  //check in Phi

             Cluster cc(*((Cluster*)c2));
             cc.GoToFrameTrk();
             Float_t xyz3[3]; c3->GetGlobalXYZ(xyz3);
             AddCookedSeed(xyz1, kSeedingLayer1, n1,
                           xyz3, kSeedingLayer3, n3, 
                           &cc,  kSeedingLayer2, n2);

	 }
     }
   }

   for (Int_t n1=0; n1<nClusters1; n1++) {
     Cluster *c1=layer1.GetCluster(n1);
     ((Cluster*)c1)->GoToFrameTrk();
   }
   for (Int_t n2=0; n2<nClusters2; n2++) {
     Cluster *c2=layer2.GetCluster(n2);
     ((Cluster*)c2)->GoToFrameTrk();
   }
   for (Int_t n3=0; n3<nClusters3; n3++) {
     Cluster *c3=layer3.GetCluster(n3);
     ((Cluster*)c3)->GoToFrameTrk();
   }

   return fSeeds->GetEntriesFast();
}

void CookedTracker::LoopOverSeeds(Int_t idx[], Int_t n) {
  //--------------------------------------------------------------------
  // Loop over a subset of track seeds
  //--------------------------------------------------------------------
  AliITSUthreadData *data = new AliITSUthreadData[kSeedingLayer2];

  for (Int_t i=0; i<n; i++) {
      Int_t s=idx[i];
      const CookedTrack *track=(CookedTrack*)fSeeds->At(s);

      Double_t x=track->GetX();
      Double_t y=track->GetY();
      Double_t phi=track->GetAlpha() + TMath::ATan2(y,x);
      const Float_t pi2 = 2.*TMath::Pi();
      if (phi<0.) phi+=pi2;
      else if (phi >= pi2) phi-=pi2;

      Double_t z=track->GetZ();
      Double_t crv=track->GetC(GetBz());
      Double_t tgl=track->GetTgl();
      Double_t r1=fgLayers[kSeedingLayer2].GetR();

      for (Int_t l=kSeedingLayer2-1; l>=0; l--) {
        Double_t r2=fgLayers[l].GetR();
        phi += 0.5*crv*(r2-r1);
        z += tgl/(0.5*crv)*(TMath::ASin(0.5*crv*r2) - TMath::ASin(0.5*crv*r1)); 
        data[l].Nsel()=0;
        data[l].ResetSelectedClusters();
        fgLayers[l].SelectClusters
           (data[l].Nsel(),data[l].Index(), phi, kRoadY, z, kRoadZ);
        r1=r2;
      }

      CookedTrack *best = new CookedTrack(*track);

      Int_t volID=-1;
      Int_t ci=-1;
      CookedTrack t3(*track);
      while ((ci=data[3].GetNextClusterIndex()) >= 0) {
        if (!AttachCluster(volID, 3, ci, t3, *track)) continue;

	CookedTrack t2(t3);
        while ((ci=data[2].GetNextClusterIndex()) >= 0) {
	  if (!AttachCluster(volID, 2, ci, t2, t3)) continue;

	  CookedTrack t1(t2);
          while ((ci=data[1].GetNextClusterIndex()) >= 0) {
            if (!AttachCluster(volID, 1, ci, t1, t2)) continue;

	    CookedTrack t0(t1);
            while ((ci=data[0].GetNextClusterIndex()) >= 0) {
              if (!AttachCluster(volID, 0, ci, t0, t1)) continue;
              if (t0.IsBetter(best,kmaxChi2PerTrack)) {
		 delete best;
                 best=new CookedTrack(t0);
	      }
              volID=-1;
	    }
            data[0].ResetSelectedClusters();
	  }
          data[1].ResetSelectedClusters();
        }
        data[2].ResetSelectedClusters();
      }

      if (best->GetNumberOfClusters() >= kminNumberOfClusters) {
	//UseClusters(best);
        Int_t noc=best->GetNumberOfClusters();
        for (Int_t i=3; i<noc; i++) {
          Int_t index=best->GetClusterIndex(i);
          Int_t l=(index & 0xf0000000) >> 28, c=(index & 0x0fffffff);
          data[l].UseCluster(c);
	}
      }
      delete fSeeds->RemoveAt(s);
      fSeeds->AddAt(best,s);
  }
  delete[] data;
}


Int_t CookedTracker::Clusters2Tracks(std::vector *event) {
  //--------------------------------------------------------------------
  // This is the main tracking function
  // The clusters must already be loaded
  //--------------------------------------------------------------------

  if (!fSAonly) AliITSUTrackerGlo::Clusters2Tracks(event);

  if (fSeeds) {fSeeds->Delete(); delete fSeeds;}
  fSeeds=new TObjArray(77777);

  //Seeding with the triggered primary vertex
  Double_t xyz[3];
  const AliESDVertex *vtx=0;
  vtx=event->GetPrimaryVertexSPD();
  if (vtx->GetStatus()) {
     xyz[0]=vtx->GetX(); xyz[1]=vtx->GetY(); xyz[2]=vtx->GetZ();
     SetVertex(xyz);
     MakeSeeds();
  }
  //Seeding with the pileup primary vertices
  TClonesArray *verticesSPD=event->GetPileupVerticesSPD();
  Int_t nfoundSPD=verticesSPD->GetEntries(); 
  for (Int_t v=0; v<nfoundSPD; v++) {
      vtx=(AliESDVertex *)verticesSPD->UncheckedAt(v);
      if (!vtx->GetStatus()) continue;
      xyz[0]=vtx->GetX(); xyz[1]=vtx->GetY(); xyz[2]=vtx->GetZ();
      SetVertex(xyz);
      MakeSeeds();
  }
  fSeeds->Sort();
  Int_t nSeeds=fSeeds->GetEntriesFast();
  Info("Clusters2Tracks","Seeds: %d",nSeeds);


#ifdef _OPENMP
   Int_t nThreads=1;

   #pragma omp parallel
   nThreads=omp_get_num_threads();

   Int_t *idx = new Int_t[nThreads*nSeeds];
   Int_t *n   = new Int_t[nThreads];
   for (Int_t i=0; i<nThreads; i++) n[i]=0;

   for (Int_t i=0; i<nSeeds; i++) {
      CookedTrack *track=(CookedTrack*)fSeeds->At(i);

      Int_t noc=track->GetNumberOfClusters();
      Int_t ci=track->GetClusterIndex(noc-1);
      Int_t l=(ci & 0xf0000000) >> 28, c=(ci & 0x0fffffff);
      Float_t phi=fgLayers[l].GetClusterPhi(c);
      Float_t bin=2*TMath::Pi()/nThreads;

      bin=2*(2*TMath::Pi())/nThreads;
      if (track->GetZ()>0) phi+=2*TMath::Pi();

      Int_t id=phi/bin;

      idx[id*nSeeds + n[id]] = i;
      n[id]++;
    }

   #pragma omp parallel
   {
   Int_t id=omp_get_thread_num();

   Float_t f=n[id]/Float_t(nSeeds);
   #pragma omp critical
   Info("Clusters2Tracks","ThreadID %d  Seed fraction %f", id, f);

   LoopOverSeeds((idx + id*nSeeds),n[id]);
   }

   delete[] idx;
   delete[] n;
#else
   Int_t *idx=new Int_t[nSeeds];
   for (Int_t i=0; i<nSeeds; i++) idx[i]=i;
   LoopOverSeeds(idx,nSeeds);
   delete[] idx;
#endif

  Int_t ngood=0;
  for (Int_t s=0; s<nSeeds; s++) {
      CookedTrack *track=(CookedTrack*)fSeeds->At(s);

      if (track->GetNumberOfClusters() < kminNumberOfClusters) continue;

      CookLabel(track,0.); //For comparison only
      Int_t label=track->GetLabel();
      if (label>0) ngood++;

      AliESDtrack iotrack;
      iotrack.UpdateTrackParams(track,AliESDtrack::kITSin);
      iotrack.SetLabel(label);
      if (fSAonly) iotrack.SetStatus(AliESDtrack::kITSpureSA); 
      event->AddTrack(&iotrack);
  }

  if (nSeeds)
  Info("Clusters2Tracks","Good tracks/seeds: %f",Float_t(ngood)/nSeeds);

  if (fSeeds) {fSeeds->Delete(); delete fSeeds;}
  fSeeds=0;
    
  return 0;
}

Int_t CookedTracker::PropagateBack(std::vector *event) {
  //--------------------------------------------------------------------
  // Here, we implement the Kalman smoother ?
  // The clusters must already be loaded
  //--------------------------------------------------------------------
  Int_t n=event->GetNumberOfTracks();
  Int_t ntrk=0;
  Int_t ngood=0;

#pragma omp parallel for reduction (+:ntrk,ngood)
  for (Int_t i=0; i<n; i++) {
      AliESDtrack *esdTrack=event->GetTrack(i);

      if (!esdTrack->IsOn(AliESDtrack::kITSin)) continue;
      if ( esdTrack->IsOn(AliESDtrack::kTPCin)) continue;//skip a TPC+ITS track

      CookedTrack track(*esdTrack);
      CookedTrack toRefit(track);

      toRefit.ResetCovariance(10.); toRefit.ResetClusters();
      if (RefitAt(40., &toRefit, &track)) {

         CookLabel(&toRefit, 0.); //For comparison only
         Int_t label=toRefit.GetLabel();
         if (label>0) ngood++;

         esdTrack->UpdateTrackParams(&toRefit,AliESDtrack::kITSout);
         ntrk++;
      }
  }

  Info("PropagateBack","Back propagated tracks: %d",ntrk);
  if (ntrk)
  Info("PropagateBack","Good tracks/back propagated: %f",Float_t(ngood)/ntrk);
  
  if (!fSAonly) AliITSUTrackerGlo::PropagateBack(event);
  
  return 0;
}

Bool_t CookedTracker::
RefitAt(Double_t xx, CookedTrack *t, const CookedTrack *c) {
  //--------------------------------------------------------------------
  // This function refits the track "t" at the position "x" using
  // the clusters from "c"
  //--------------------------------------------------------------------
  Int_t index[kNLayers];
  Int_t k;
  for (k=0; k<kNLayers; k++) index[k]=-1;
  Int_t nc=c->GetNumberOfClusters();
  for (k=0; k<nc; k++) {
    Int_t idx=c->GetClusterIndex(k), nl=(idx&0xf0000000)>>28;
    index[nl]=idx;
  }

  Int_t from, to, step;
  if (xx > t->GetX()) {
      from=0; to=kNLayers;
      step=+1;
  } else {
      from=kNLayers-1; to=-1;
      step=-1;
  }

  for (Int_t i=from; i != to; i += step) {
     Int_t idx=index[i];
     if (idx>=0) {
        const Cluster *cl=GetCluster(idx);
        Float_t xr,ar; cl->GetXAlphaRefPlane(xr, ar);
        if (!t->Propagate(Double_t(ar), Double_t(xr), GetBz())) {
           //Warning("RefitAt","propagation failed !\n");
           return kFALSE;
        }
        Double_t chi2=t->GetPredictedChi2(cl);
        if (chi2 < kmaxChi2PerCluster) t->Update(cl, chi2, idx);
     } else {
        Double_t r=fgLayers[i].GetR();
        Double_t phi,z;
        if (!t->GetPhiZat(r,phi,z)) {
           //Warning("RefitAt","failed to estimate track !\n");
           return kFALSE;
        }
        if (!t->Propagate(phi, r, GetBz())) {
           //Warning("RefitAt","propagation failed !\n");
           return kFALSE;
        }
     }
     Double_t xx0 = (i > 2) ? 0.008 : 0.003;  // Rough layer thickness
     Double_t x0  = 9.36; // Radiation length of Si [cm]
     Double_t rho = 2.33; // Density of Si [g/cm^3]
     Double_t mass = t->GetMass();
     t->CorrectForMeanMaterial(xx0, -step*xx0*x0*rho, mass, kTRUE);
  }

  if (!t->PropagateTo(xx,0.,0.)) return kFALSE;
  return kTRUE;
}

Int_t CookedTracker::RefitInward(std::vector *event) {
  //--------------------------------------------------------------------
  // Some final refit, after the outliers get removed by the smoother ?  
  // The clusters must be loaded
  //--------------------------------------------------------------------
  Int_t n=event->GetNumberOfTracks();
  Int_t ntrk=0;
  Int_t ngood=0;

#pragma omp parallel for reduction (+:ntrk,ngood)
  for (Int_t i=0; i<n; i++) {
      AliESDtrack *esdTrack=event->GetTrack(i);

      if (!esdTrack->IsOn(AliESDtrack::kITSout)) continue;
      if ( esdTrack->IsOn(AliESDtrack::kTPCin)) continue;//skip a TPC+ITS track

      CookedTrack track(*esdTrack);
      CookedTrack toRefit(track);

      toRefit.ResetCovariance(10.); toRefit.ResetClusters();
      if (!RefitAt(2.1, &toRefit, &track)) continue;
      //Cross the beam pipe
      if (!toRefit.PropagateTo(1.8, 2.27e-3, 35.28*1.848)) continue;

      CookLabel(&toRefit, 0.); //For comparison only
      Int_t label=toRefit.GetLabel();
      if (label>0) ngood++;

      esdTrack->UpdateTrackParams(&toRefit,AliESDtrack::kITSrefit);
      //esdTrack->RelateToVertex(event->GetVertex(),GetBz(),33.);
      ntrk++;
  }

  Info("RefitInward","Refitted tracks: %d",ntrk);
  if (ntrk)
  Info("RefitInward","Good tracks/refitted: %f",Float_t(ngood)/ntrk);
    
  if (!fSAonly) AliITSUTrackerGlo::RefitInward(event);

  return 0;
}

Int_t CookedTracker::LoadClusters(TTree *cTree) {
  //--------------------------------------------------------------------
  // This function reads the ITSU clusters from the tree,
  // sort them, distribute over the internal tracker arrays, etc
  //--------------------------------------------------------------------
  if (!cTree) {
     AliFatal("No cluster tree !");
     return 1;
  }

  AliITSUTrackerGlo::LoadClusters(cTree);

  Bool_t glo[kNLayers]={0};
  glo[kSeedingLayer1]=glo[kSeedingLayer2]=glo[kSeedingLayer3]=kTRUE;
#pragma omp parallel for
  for (Int_t i=0; i<kNLayers; i++) {
      TClonesArray *clusters=fReconstructor->GetClusters(i);
      fgLayers[i].InsertClusters(clusters,glo[i],fSAonly);
  }

  return 0;
}

void CookedTracker::UnloadClusters() {
  //--------------------------------------------------------------------
  // This function unloads ITSU clusters from the RAM
  //--------------------------------------------------------------------
  AliITSUTrackerGlo::UnloadClusters();
  for (Int_t i=0; i<kNLayers; i++) fgLayers[i].DeleteClusters();
}

Cluster *CookedTracker::GetCluster(Int_t index) const {
  //--------------------------------------------------------------------
  //       Return pointer to a given cluster
  //--------------------------------------------------------------------
    Int_t l=(index & 0xf0000000) >> 28;
    Int_t c=(index & 0x0fffffff) >> 00;
    return fgLayers[l].GetCluster(c);
}

CookedTracker::AliITSUlayer::AliITSUlayer():
  fR(0),
  fN(0)
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
  for (Int_t i=0; i<kMaxClusterPerLayer; i++) fClusters[i]=0;
}

CookedTracker::AliITSUthreadData::AliITSUthreadData()
  :fNsel(0)
  ,fI(0)
{
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
  for (Int_t i=0; i<kMaxClusterPerLayer; i++) fUsed[i]=kFALSE;
}

void CookedTracker::AliITSUlayer::
InsertClusters(TClonesArray *clusters, Bool_t seedingLayer, Bool_t saOnly)
{
  //--------------------------------------------------------------------
  // Load clusters to this layer
  //--------------------------------------------------------------------
  Int_t ncl=clusters->GetEntriesFast();
  Double_t r=0.;
  for (Int_t i=0; i<ncl; i++) {
     Cluster *c=(Cluster*)clusters->UncheckedAt(i);
     if (!saOnly) if (c->IsClusterUsed()) continue;
     c->GoToFrameGlo();
     Double_t x=c->GetX(), y=c->GetY();
     r += TMath::Sqrt(x*x + y*y);
     if (!seedingLayer) c->GoToFrameTrk();
     //if (!c->Misalign()) AliWarning("Can't misalign this cluster !");
     if (InsertCluster(c)) break;
  }
  if (fN) fR = r/fN;
  const Float_t pi2 = 2.*TMath::Pi();
  for (Int_t i=0; i<fN; i++) {
      Cluster *c=fClusters[i];
      c->GetXAlphaRefPlane(fXRef[i],fAlphaRef[i]);
      Float_t xyz[3]; c->GetGlobalXYZ(xyz);
      Float_t phi=TMath::ATan2(xyz[1],xyz[0]);
      if (phi<0.) phi+=pi2;
      else if (phi >= pi2) phi-=pi2;
      fPhi[i]=phi;
  }
}

void CookedTracker::AliITSUlayer::DeleteClusters()
{
  //--------------------------------------------------------------------
  // Load clusters to this layer
  //--------------------------------------------------------------------
  //for (Int_t i=0; i<fN; i++) {delete fClusters[i]; fClusters[i]=0;}
  fN=0;
}

Int_t 
CookedTracker::AliITSUlayer::InsertCluster(Cluster *c) {
  //--------------------------------------------------------------------
  // This function inserts a cluster to this layer in increasing
  // order of the cluster's fZ
  //--------------------------------------------------------------------
  if (fN>=kMaxClusterPerLayer) {
     ::Error("InsertCluster","Too many clusters !\n");
     return 1;
  }
  if (fN==0) fClusters[0]=c;
  else {
     Int_t i=FindClusterIndex(c->GetZ());
     Int_t k=fN-i;
     memmove(fClusters+i+1 ,fClusters+i,k*sizeof(Cluster*));
     fClusters[i]=c;
  }
  fN++;
  return 0;
}

Int_t 
CookedTracker::AliITSUlayer::FindClusterIndex(Double_t z) const {
  //--------------------------------------------------------------------
  // This function returns the index of the first 
  // with its fZ >= "z". 
  //--------------------------------------------------------------------
  if (fN==0) return 0;

  Int_t b=0;
  if (z <= fClusters[b]->GetZ()) return b;

  Int_t e=b+fN-1;
  if (z > fClusters[e]->GetZ()) return e+1;

  Int_t m=(b+e)/2;
  for (; b<e; m=(b+e)/2) {
    if (z > fClusters[m]->GetZ()) b=m+1;
    else e=m; 
  }
  return m;
}

void CookedTracker::AliITSUlayer::SelectClusters
(Int_t &n, Int_t idx[], Float_t phi, Float_t dy, Float_t z, Float_t dz) {
  //--------------------------------------------------------------------
  // This function selects clusters within the "road"
  //--------------------------------------------------------------------
  Float_t dphi=dy/fR; 
  Float_t phiMin=phi-dphi;
  Float_t phiMax=phi+dphi;
  Float_t zMin=z-dz;
  Float_t zMax=z+dz;
 
  Int_t imin=FindClusterIndex(zMin), imax=FindClusterIndex(zMax);
  for (Int_t i=imin; i<imax; i++) {
      Float_t cphi=fPhi[i];
      if (cphi <= phiMin) continue;
      if (cphi >  phiMax) continue;

      idx[n++]=i;
      if (n >= kMaxSelected) return;
  } 

}

Bool_t
CookedTracker::AttachCluster
(Int_t &volID, Int_t nl, Int_t ci, AliKalmanTrack &t, const AliKalmanTrack &o) const {

   AliITSUlayer &layer=fgLayers[nl];
   Cluster *c=layer.GetCluster(ci);

   Int_t vid=c->GetVolumeId();

   if (vid != volID) {
      volID=vid;
      t=o;
      Double_t x=layer.GetXRef(ci);
      Double_t alpha=layer.GetAlphaRef(ci);
      if (!t.Propagate(alpha, x, GetBz())) return kFALSE;
   }

   Double_t chi2=t.GetPredictedChi2(c);
   if (chi2 > kmaxChi2PerCluster) return kFALSE;

   if ( !t.Update(c,chi2,(nl<<28)+ci) ) return kFALSE;

   Double_t xx0 = (nl > 2) ? 0.008 : 0.003;  // Rough layer thickness
   Double_t x0  = 9.36; // Radiation length of Si [cm]
   Double_t rho = 2.33; // Density of Si [g/cm^3]
   Double_t mass = t.GetMass();
   t.CorrectForMeanMaterial(xx0, xx0*x0*rho, mass, kTRUE);

   return kTRUE;
}
