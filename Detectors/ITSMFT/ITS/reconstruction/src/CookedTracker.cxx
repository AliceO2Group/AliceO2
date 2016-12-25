/// \file CookedTracker.cxx
/// \brief Implementation of the "Cooked Matrix" ITS tracker
/// \author iouri.belikov@cern.ch

//-------------------------------------------------------------------------
//                     A stand-alone ITS tracker
//    The pattern recongintion based on the "cooked covariance" approach
//-------------------------------------------------------------------------

#include <TClonesArray.h>
#include <TGeoGlobalMagField.h>
#include <TMath.h>

#include "FairLogger.h"

#include "DetectorsBase/Constants.h"
#include "Field/MagneticField.h"
#include "ITSReconstruction/Cluster.h"
#include "ITSReconstruction/CookedTrack.h"
#include "ITSReconstruction/CookedTracker.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace AliceO2::ITS;
using namespace AliceO2::Base::Constants;
using AliceO2::Field::MagneticField;

//************************************************
// Constants hardcoded for the moment:
//************************************************
// seed "windows" in z and phi: makeSeeds
const Double_t kzWin = 0.33;
const Double_t kminPt = 0.05;
// Maximal accepted impact parameters for the seeds
const Double_t kmaxDCAxy = 3.;
const Double_t kmaxDCAz = 3.;
// Layers for the seeding
const Int_t kSeedingLayer1 = 6, kSeedingLayer2 = 4, kSeedingLayer3 = 5;
// Space point resolution
const Double_t kSigma2 = 0.0005 * 0.0005;
// Max accepted chi2
const Double_t kmaxChi2PerCluster = 20.;
const Double_t kmaxChi2PerTrack = 30.;
// Tracking "road" from layer to layer
const Double_t kRoadY = 0.2;
const Double_t kRoadZ = 0.7;
// Minimal number of attached clusters
const Int_t kminNumberOfClusters = 4;

//************************************************
// TODO:
//************************************************
// Seeding:
// Precalculate cylidnrical (r,phi) for the clusters;
// use exact r's for the clusters

CookedTracker::Layer CookedTracker::sLayers[CookedTracker::kNLayers];

CookedTracker::CookedTracker()
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
  const Double_t klRadius[7] = { 2.34, 3.15, 3.93, 19.61, 24.55, 34.39, 39.34 }; // tdr6

  for (Int_t i = 0; i < kNLayers; i++)
    sLayers[i].setR(klRadius[i]);

  // Some default primary vertex
  Double_t xyz[] = { 0., 0., 0. };
  Double_t ers[] = { 2., 2., 2. };

  setVertex(xyz, ers);
}

CookedTracker::~CookedTracker()
{
  //--------------------------------------------------------------------
  // Virtual destructor
  //--------------------------------------------------------------------
}

//__________________________________________________________________________
void CookedTracker::cookLabel(CookedTrack& t, Float_t wrong) const
{
  //--------------------------------------------------------------------
  // This function "cooks" a track label.
  // A label<0 indicates that some of the clusters are wrongly assigned.
  //--------------------------------------------------------------------
  Int_t noc = t.getNumberOfClusters();
  if (noc < 1)
    return;
  std::vector<Int_t> lb(noc, 0);
  std::vector<Int_t> mx(noc, 0);
  std::vector<Cluster*> clusters(noc);

  Int_t i;
  for (i = 0; i < noc; i++) {
    Int_t index = t.getClusterIndex(i);
    clusters[i] = getCluster(index);
  }

  Int_t lab = 123456789;
  for (i = 0; i < noc; i++) {
    Cluster* c = clusters[i];
    lab = TMath::Abs(c->getLabel(0));
    Int_t j;
    for (j = 0; j < noc; j++)
      if (lb[j] == lab || mx[j] == 0)
        break;
    if (j < noc) {
      lb[j] = lab;
      (mx[j])++;
    }
  }

  Int_t max = 0;
  for (i = 0; i < noc; i++)
    if (mx[i] > max) {
      max = mx[i];
      lab = lb[i];
    }

  for (i = 0; i < noc; i++) {
    Cluster* c = clusters[i];
    // if (TMath::Abs(c->getLabel(1)) == lab ||
    //    TMath::Abs(c->getLabel(2)) == lab ) max++;
    if (TMath::Abs(c->getLabel(0) != lab))
      if (TMath::Abs(c->getLabel(1)) == lab || TMath::Abs(c->getLabel(2)) == lab)
        max++;
  }

  if ((1. - Float_t(max) / noc) > wrong)
    lab = -lab;
  // t.SetFakeRatio((1.- Float_t(max)/noc));
  t.setLabel(lab);
}

Double_t CookedTracker::getBz()
{
  MagneticField* fld = (MagneticField*)TGeoGlobalMagField::Instance()->GetField();
  if (!fld) {
    LOG(FATAL) << "Field is not loaded !" << FairLogger::endl;
    return kAlmost0;
  }
  Double_t bz = fld->solenoidField();
  return TMath::Sign(kAlmost0, bz) + bz;
}

static Double_t f1(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t x3, Double_t y3)
{
  //-----------------------------------------------------------------
  // Initial approximation of the track curvature
  //-----------------------------------------------------------------
  Double_t d = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1);
  Double_t a =
    0.5 * ((y3 - y2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1) - (y2 - y1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2));
  Double_t b =
    0.5 * ((x2 - x1) * (y3 * y3 - y2 * y2 + x3 * x3 - x2 * x2) - (x3 - x2) * (y2 * y2 - y1 * y1 + x2 * x2 - x1 * x1));

  Double_t xr = TMath::Abs(d / (d * x1 - a)), yr = TMath::Abs(d / (d * y1 - b));

  Double_t crv = xr * yr / sqrt(xr * xr + yr * yr);
  if (d > 0)
    crv = -crv;

  return crv;
}

static Double_t f2(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t x3, Double_t y3)
{
  //-----------------------------------------------------------------
  // Initial approximation of the x-coordinate of the center of curvature
  //-----------------------------------------------------------------

  Double_t k1 = (y2 - y1) / (x2 - x1), k2 = (y3 - y2) / (x3 - x2);
  Double_t x0 = 0.5 * (k1 * k2 * (y1 - y3) + k2 * (x1 + x2) - k1 * (x2 + x3)) / (k2 - k1);

  return x0;
}

static Double_t f3(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t z1, Double_t z2)
{
  //-----------------------------------------------------------------
  // Initial approximation of the tangent of the track dip angle
  //-----------------------------------------------------------------
  return (z1 - z2) / sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

Bool_t CookedTracker::addCookedSeed(const Float_t r1[3], Int_t l1, Int_t i1, const Float_t r2[3], Int_t l2, Int_t i2,
                                    const Cluster* c3, Int_t l3, Int_t i3)
{
  //--------------------------------------------------------------------
  // This is the main cooking function.
  // Creates seed parameters out of provided clusters.
  //--------------------------------------------------------------------
  Float_t x, a;
  if (!c3->getXAlphaRefPlane(x, a))
    return kFALSE;

  Double_t ca = TMath::Cos(a), sa = TMath::Sin(a);
  Double_t x1 = r1[0] * ca + r1[1] * sa, y1 = -r1[0] * sa + r1[1] * ca, z1 = r1[2];
  Double_t x2 = r2[0] * ca + r2[1] * sa, y2 = -r2[0] * sa + r2[1] * ca, z2 = r2[2];
  Double_t x3 = x, y3 = c3->getY(), z3 = c3->getZ();

  Float_t par[5];
  par[0] = y3;
  par[1] = z3;
  Double_t crv = f1(x1, y1, x2, y2, x3, y3); // curvature
  Double_t x0 = f2(x1, y1, x2, y2, x3, y3);  // x-coordinate of the center
  Double_t tgl12 = f3(x1, y1, x2, y2, z1, z2);
  Double_t tgl23 = f3(x2, y2, x3, y3, z2, z3);

  Double_t sf = crv * (x - x0);
  if (TMath::Abs(sf) >= kAlmost1)
    return kFALSE;
  par[2] = sf;

  par[3] = 0.5 * (tgl12 + tgl23);
  Double_t bz = getBz();
  par[4] = (TMath::Abs(bz) < kAlmost0) ? kAlmost0 : crv / (bz * kB2C);

  Float_t cov[15];
  /*
  for (Int_t i=0; i<15; i++) cov[i]=0.;
  cov[0] =kSigma2*10;
  cov[2] =kSigma2*10;
  cov[5] =0.007*0.007*10;   //FIXME all these lines
  cov[9] =0.007*0.007*10;
  cov[14]=0.1*0.1*10;
  */
  const Double_t dlt = 0.0005;
  Double_t fy = 1. / (sLayers[kSeedingLayer3].getR() - sLayers[kSeedingLayer2].getR());
  Double_t tz = fy;
  Double_t cy = (f1(x1, y1, x2, y2 + dlt, x3, y3) - crv) / dlt / bz / kB2C;
  cy *= 20; // FIXME: MS contribution to the cov[14]
  Double_t s2 = kSigma2;

  cov[0] = s2;
  cov[1] = 0.;
  cov[2] = s2;
  cov[3] = s2 * fy;
  cov[4] = 0.;
  cov[5] = s2 * fy * fy;
  cov[6] = 0.;
  cov[7] = s2 * tz;
  cov[8] = 0.;
  cov[9] = s2 * tz * tz;
  cov[10] = s2 * cy;
  cov[11] = 0.;
  cov[12] = s2 * fy * cy;
  cov[13] = 0.;
  cov[14] = s2 * cy * cy;

  CookedTrack seed(x, a, par, cov);

  Double_t dz[2];
  seed.getImpactParams(getX(), getY(), getZ(), getBz(), dz);
  if (TMath::Abs(dz[0]) > kmaxDCAxy)
    return kFALSE;
  if (TMath::Abs(dz[1]) > kmaxDCAz)
    return kFALSE;

  Double_t xx0 = 0.008; // Rough layer thickness
  Double_t radl = 9.36; // Radiation length of Si [cm]
  Double_t rho = 2.33;  // Density of Si [g/cm^3]
  if (!seed.correctForMeanMaterial(xx0, xx0 * radl * rho, kTRUE))
    return kFALSE;

  seed.setClusterIndex(l1, i1);
  seed.setClusterIndex(l2, i2);
  seed.setClusterIndex(l3, i3);

#pragma omp critical
  mSeeds.push_back(seed);

  return kTRUE;
}

Int_t CookedTracker::makeSeeds()
{
  //--------------------------------------------------------------------
  // This is the main pattern recongition function.
  // Creates seeds out of two clusters and another point.
  //--------------------------------------------------------------------
  const Double_t zv = getZ();

  Layer& layer1 = sLayers[kSeedingLayer1];
  Layer& layer2 = sLayers[kSeedingLayer2];
  Layer& layer3 = sLayers[kSeedingLayer3];

  const Double_t maxC = TMath::Abs(getBz() * kB2C / kminPt);
  const Double_t kpWin = TMath::ASin(0.5 * maxC * layer1.getR()) - TMath::ASin(0.5 * maxC * layer2.getR());

  Int_t nClusters1 = layer1.getNumberOfClusters();
  Int_t nClusters2 = layer2.getNumberOfClusters();
  Int_t nClusters3 = layer3.getNumberOfClusters();
#pragma omp parallel for
  for (Int_t n1 = 0; n1 < nClusters1; n1++) {
    Cluster* c1 = layer1.getCluster(n1);
    //
    // Int_t lab=c1->getLabel(0);
    //
    Double_t z1 = c1->getZ();
    Float_t xyz1[3];
    c1->getGlobalXYZ(xyz1);
    Double_t r1 = TMath::Sqrt(xyz1[0] * xyz1[0] + xyz1[1] * xyz1[1]);
    Double_t phi1 = layer1.getClusterPhi(n1);

    Double_t zr2 = zv + layer2.getR() / r1 * (z1 - zv);
    Int_t start2 = layer2.findClusterIndex(zr2 - kzWin);
    for (Int_t n2 = start2; n2 < nClusters2; n2++) {
      Cluster* c2 = layer2.getCluster(n2);
      //
      // if (c2->getLabel(0)!=lab) continue;
      //
      Double_t z2 = c2->getZ();
      if (z2 > (zr2 + kzWin))
        break; // check in Z

      Double_t phi2 = layer2.getClusterPhi(n2);
      if (TMath::Abs(phi2 - phi1) > kpWin)
        continue; // check in Phi

      Float_t xyz2[3];
      c2->getGlobalXYZ(xyz2);
      Double_t r2 = TMath::Sqrt(xyz2[0] * xyz2[0] + xyz2[1] * xyz2[1]);
      Double_t crv = f1(xyz1[0], xyz1[1], xyz2[0], xyz2[1], getX(), getY());

      Double_t zr3 = z1 + (layer3.getR() - r1) / (r2 - r1) * (z2 - z1);
      Double_t dz = kzWin / 2;
      Int_t start3 = layer3.findClusterIndex(zr3 - dz);
      for (Int_t n3 = start3; n3 < nClusters3; n3++) {
        Cluster* c3 = layer3.getCluster(n3);
        //
        // if (c3->getLabel(0)!=lab) continue;
        //
        Double_t z3 = c3->getZ();
        if (z3 > (zr3 + dz))
          break; // check in Z

        Double_t r3 = layer3.getXRef(n3);
        Double_t phir3 = phi1 + 0.5 * crv * (r3 - r1);
        Double_t phi3 = layer3.getClusterPhi(n3);
        if (TMath::Abs(phir3 - phi3) > kpWin / 100)
          continue; // check in Phi

        Cluster cc(*((Cluster*)c2));
        cc.goToFrameTrk();
        Float_t xyz3[3];
        c3->getGlobalXYZ(xyz3);
        addCookedSeed(xyz1, kSeedingLayer1, n1, xyz3, kSeedingLayer3, n3, &cc, kSeedingLayer2, n2);
      }
    }
  }

  for (Int_t n1 = 0; n1 < nClusters1; n1++) {
    Cluster* c1 = layer1.getCluster(n1);
    ((Cluster*)c1)->goToFrameTrk();
  }
  for (Int_t n2 = 0; n2 < nClusters2; n2++) {
    Cluster* c2 = layer2.getCluster(n2);
    ((Cluster*)c2)->goToFrameTrk();
  }
  for (Int_t n3 = 0; n3 < nClusters3; n3++) {
    Cluster* c3 = layer3.getCluster(n3);
    ((Cluster*)c3)->goToFrameTrk();
  }

  return mSeeds.size();
}

void CookedTracker::loopOverSeeds(Int_t idx[], Int_t n)
{
  //--------------------------------------------------------------------
  // Loop over a subset of track seeds
  //--------------------------------------------------------------------
  ThreadData* data = new ThreadData[kSeedingLayer2];

  for (Int_t i = 0; i < n; i++) {
    Int_t s = idx[i];
    const CookedTrack& track = mSeeds[s];

    Double_t x = track.getX();
    Double_t y = track.getY();
    Double_t phi = track.getAlpha() + TMath::ATan2(y, x);
    const Float_t pi2 = 2. * TMath::Pi();
    if (phi < 0.)
      phi += pi2;
    else if (phi >= pi2)
      phi -= pi2;

    Double_t z = track.getZ();
    Double_t crv = track.getCurvature(getBz());
    Double_t tgl = track.getTgl();
    Double_t r1 = sLayers[kSeedingLayer2].getR();

    for (Int_t l = kSeedingLayer2 - 1; l >= 0; l--) {
      Double_t r2 = sLayers[l].getR();
      phi += 0.5 * crv * (r2 - r1);
      z += tgl / (0.5 * crv) * (TMath::ASin(0.5 * crv * r2) - TMath::ASin(0.5 * crv * r1));
      data[l].Nsel() = 0;
      data[l].resetSelectedClusters();
      sLayers[l].selectClusters(data[l].Nsel(), data[l].Index(), phi, kRoadY, z, kRoadZ);
      r1 = r2;
    }

    CookedTrack best(track);

    Int_t volID = -1;
    Int_t ci = -1;
    CookedTrack t3(track);
    while ((ci = data[3].getNextClusterIndex()) >= 0) {
      if (!attachCluster(volID, 3, ci, t3, track))
        continue;

      CookedTrack t2(t3);
      while ((ci = data[2].getNextClusterIndex()) >= 0) {
        if (!attachCluster(volID, 2, ci, t2, t3))
          continue;

        CookedTrack t1(t2);
        while ((ci = data[1].getNextClusterIndex()) >= 0) {
          if (!attachCluster(volID, 1, ci, t1, t2))
            continue;

          CookedTrack t0(t1);
          while ((ci = data[0].getNextClusterIndex()) >= 0) {
            if (!attachCluster(volID, 0, ci, t0, t1))
              continue;
            if (t0.isBetter(best, kmaxChi2PerTrack)) {
              best = t0;
            }
            volID = -1;
          }
          data[0].resetSelectedClusters();
        }
        data[1].resetSelectedClusters();
      }
      data[2].resetSelectedClusters();
    }

    if (best.getNumberOfClusters() >= kminNumberOfClusters) {
      // useClusters(best);
      Int_t noc = best.getNumberOfClusters();
      for (Int_t ic = 3; ic < noc; ic++) {
        Int_t index = best.getClusterIndex(ic);
        Int_t l = (index & 0xf0000000) >> 28, c = (index & 0x0fffffff);
        data[l].useCluster(c);
      }
    }
    mSeeds[s] = best;
  }
  delete[] data;
}

void CookedTracker::process(const TClonesArray& clusters, TClonesArray& tracks)
{
  //--------------------------------------------------------------------
  // This is the main tracking function
  //--------------------------------------------------------------------

  loadClusters(clusters);

  mSeeds.clear();

  // Seeding with the triggered primary vertex
  Double_t xyz[3]{ 0, 0, 0 }; // FIXME
  setVertex(xyz);
  makeSeeds();

  // Seeding with the pileup primary vertices
  /* FIXME
  TClonesArray *verticesSPD=tracks->GetPileupVerticesSPD();
  Int_t nfoundSPD=verticesSPD->GetEntries();
  for (Int_t v=0; v<nfoundSPD; v++) {
      vtx=(AliESDVertex *)verticesSPD->UncheckedAt(v);
      if (!vtx->GetStatus()) continue;
      xyz[0]=vtx->getX(); xyz[1]=vtx->getY(); xyz[2]=vtx->getZ();
      setVertex(xyz);
      makeSeeds();
  }
  */

  std::sort(mSeeds.begin(), mSeeds.end());

  Int_t nSeeds = mSeeds.size();
  Info("Clusters2Tracks", "Seeds: %d", nSeeds);

#ifdef _OPENMP
  Int_t nThreads = 1;

#pragma omp parallel
  nThreads = omp_get_num_threads();

  Int_t* idx = new Int_t[nThreads * nSeeds];
  Int_t* n = new Int_t[nThreads];
  for (Int_t i = 0; i < nThreads; i++)
    n[i] = 0;

  for (Int_t i = 0; i < nSeeds; i++) {
    CookedTrack& track = mSeeds[i];

    Int_t noc = track.getNumberOfClusters();
    Int_t ci = track.getClusterIndex(noc - 1);
    Int_t l = (ci & 0xf0000000) >> 28, c = (ci & 0x0fffffff);
    Float_t phi = sLayers[l].getClusterPhi(c);
    Float_t bin = 2 * TMath::Pi() / nThreads;

    bin = 2 * (2 * TMath::Pi()) / nThreads;
    if (track.getZ() > 0)
      phi += 2 * TMath::Pi();

    Int_t id = phi / bin;

    idx[id * nSeeds + n[id]] = i;
    n[id]++;
  }

#pragma omp parallel
  {
    Int_t id = omp_get_thread_num();

    Float_t f = n[id] / Float_t(nSeeds);
#pragma omp critical
    Info("Clusters2Tracks", "ThreadID %d  Seed fraction %f", id, f);

    loopOverSeeds((idx + id * nSeeds), n[id]);
  }

  delete[] idx;
  delete[] n;
#else
  Int_t* idx = new Int_t[nSeeds];
  for (Int_t i = 0; i < nSeeds; i++)
    idx[i] = i;
  loopOverSeeds(idx, nSeeds);
  delete[] idx;
#endif

  Int_t ngood = 0;
  for (Int_t s = 0; s < nSeeds; s++) {
    CookedTrack& track = mSeeds[s];

    if (track.getNumberOfClusters() < kminNumberOfClusters)
      continue;

    cookLabel(track, 0.); // For comparison only
    Int_t label = track.getLabel();
    if (label > 0)
      ngood++;

    new (tracks[tracks.GetEntriesFast()]) CookedTrack(track);
  }

  if (nSeeds)
    Info("Clusters2Tracks", "Good tracks/seeds: %f", Float_t(ngood) / nSeeds);

  unloadClusters();
}

/*
Int_t CookedTracker::propagateBack(std::vector<CookedTrack> *tracks) {
  //--------------------------------------------------------------------
  // Here, we implement the Kalman smoother ?
  // The clusters must already be loaded
  //--------------------------------------------------------------------
  Int_t n=tracks->getNumberOfTracks();
  Int_t ntrk=0;
  Int_t ngood=0;

#pragma omp parallel for reduction (+:ntrk,ngood)
  for (Int_t i=0; i<n; i++) {
      CookedTrack *esdTrack=tracks->GetTrack(i);

      if (!esdTrack->IsOn(CookedTrack::kITSin)) continue;
      if ( esdTrack->IsOn(CookedTrack::kTPCin)) continue;//skip a TPC+ITS track

      CookedTrack track(*esdTrack);
      CookedTrack toRefit(track);

      toRefit.resetCovariance(10.); toRefit.resetClusters();
      if (refitAt(40., &toRefit, &track)) {

         cookLabel(toRefit, 0.); //For comparison only
         Int_t label=toRefit.getLabel();
         if (label>0) ngood++;

         esdTrack->UpdateTrackParams(&toRefit,CookedTrack::kITSout);
         ntrk++;
      }
  }

  Info("propagateBack","Back propagated tracks: %d",ntrk);
  if (ntrk)
  Info("propagateBack","Good tracks/back propagated: %f",Float_t(ngood)/ntrk);

  return 0;
}

Bool_t CookedTracker::
refitAt(Double_t xx, CookedTrack *t, const CookedTrack *c) {
  //--------------------------------------------------------------------
  // This function refits the track "t" at the position "x" using
  // the clusters from "c"
  //--------------------------------------------------------------------
  Int_t index[kNLayers];
  Int_t k;
  for (k=0; k<kNLayers; k++) index[k]=-1;
  Int_t nc=c->getNumberOfClusters();
  for (k=0; k<nc; k++) {
    Int_t idx=c->getClusterIndex(k), nl=(idx&0xf0000000)>>28;
    index[nl]=idx;
  }

  Int_t from, to, step;
  if (xx > t->getX()) {
      from=0; to=kNLayers;
      step=+1;
  } else {
      from=kNLayers-1; to=-1;
      step=-1;
  }

  for (Int_t i=from; i != to; i += step) {
     Int_t idx=index[i];
     if (idx>=0) {
        const Cluster *cl=getCluster(idx);
        Float_t xr,ar; cl->getXAlphaRefPlane(xr, ar);
        if (!t->propagate(Double_t(ar), Double_t(xr), getBz())) {
           //Warning("refitAt","propagation failed !\n");
           return kFALSE;
        }
        Double_t chi2=t->getPredictedChi2(cl);
        if (chi2 < kmaxChi2PerCluster) t->update(cl, chi2, idx);
     } else {
        Double_t r=sLayers[i].getR();
        Double_t phi,z;
        if (!t->GetPhiZat(r,phi,z)) {
           //Warning("refitAt","failed to estimate track !\n");
           return kFALSE;
        }
        if (!t->propagate(phi, r, getBz())) {
           //Warning("refitAt","propagation failed !\n");
           return kFALSE;
        }
     }
     Double_t xx0 = (i > 2) ? 0.008 : 0.003;  // Rough layer thickness
     Double_t x0  = 9.36; // Radiation length of Si [cm]
     Double_t rho = 2.33; // Density of Si [g/cm^3]
     t->correctForMeanMaterial(xx0, -step*xx0*x0*rho, kTRUE);
  }

  if (!t->propagateTo(xx,0.,0.)) return kFALSE;
  return kTRUE;
}

Int_t CookedTracker::RefitInward(std::vector<CookedTrack> *tracks) {
  //--------------------------------------------------------------------
  // Some final refit, after the outliers get removed by the smoother ?
  // The clusters must be loaded
  //--------------------------------------------------------------------
  Int_t n=tracks->getNumberOfTracks();
  Int_t ntrk=0;
  Int_t ngood=0;

#pragma omp parallel for reduction (+:ntrk,ngood)
  for (Int_t i=0; i<n; i++) {
      CookedTrack *esdTrack=tracks->GetTrack(i);

      if (!esdTrack->IsOn(CookedTrack::kITSout)) continue;
      if ( esdTrack->IsOn(CookedTrack::kTPCin)) continue;//skip a TPC+ITS track

      CookedTrack track(*esdTrack);
      CookedTrack toRefit(track);

      toRefit.resetCovariance(10.); toRefit.resetClusters();
      if (!refitAt(2.1, &toRefit, &track)) continue;
      //Cross the beam pipe
      if (!toRefit.propagateTo(1.8, 2.27e-3, 35.28*1.848)) continue;

      cookLabel(toRefit, 0.); //For comparison only
      Int_t label=toRefit.getLabel();
      if (label>0) ngood++;

      esdTrack->UpdateTrackParams(&toRefit,CookedTrack::kITSrefit);
      //esdTrack->RelateToVertex(tracks->GetVertex(),getBz(),33.);
      ntrk++;
  }

  Info("RefitInward","Refitted tracks: %d",ntrk);
  if (ntrk)
  Info("RefitInward","Good tracks/refitted: %f",Float_t(ngood)/ntrk);

  return 0;
}
*/

void CookedTracker::loadClusters(const TClonesArray& clusters)
{
  //--------------------------------------------------------------------
  // This function reads the ITSU clusters from the tree,
  // sort them, distribute over the internal tracker arrays, etc
  //--------------------------------------------------------------------
  Int_t numOfClusters = clusters.GetEntriesFast();
  if (numOfClusters == 0) {
    LOG(FATAL) << "No clusters to load !" << FairLogger::endl;
    return;
  }

#pragma omp parallel for
  for (Int_t i = 0; i < numOfClusters; i++) {
    Cluster* c = (Cluster*)clusters.UncheckedAt(i);
    c->goToFrameTrk();

    Int_t layer = c->getLayer();
    if ((layer == kSeedingLayer1) || (layer == kSeedingLayer2) || (layer == kSeedingLayer3))
      c->goToFrameGlo();

    if (!sLayers[layer].insertCluster(c))
      continue;
  }

  for (Int_t l = 0; l < kNLayers; l++)
    sLayers[l].init();
}

void CookedTracker::unloadClusters()
{
  //--------------------------------------------------------------------
  // This function unloads ITSU clusters from the RAM
  //--------------------------------------------------------------------
  for (Int_t i = 0; i < kNLayers; i++)
    sLayers[i].unloadClusters();
}

Cluster* CookedTracker::getCluster(Int_t index) const
{
  //--------------------------------------------------------------------
  //       Return pointer to a given cluster
  //--------------------------------------------------------------------
  Int_t l = (index & 0xf0000000) >> 28;
  Int_t c = (index & 0x0fffffff) >> 00;
  return sLayers[l].getCluster(c);
}

CookedTracker::Layer::Layer() : mR(0), mN(0)
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
  for (Int_t i = 0; i < kMaxClusterPerLayer; i++)
    mClusters[i] = 0;
}

CookedTracker::ThreadData::ThreadData() : mNsel(0), mI(0)
{
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
  for (Int_t i = 0; i < kMaxClusterPerLayer; i++)
    mUsed[i] = kFALSE;
}

void CookedTracker::Layer::init()
{
  //--------------------------------------------------------------------
  // Load clusters to this layer
  //--------------------------------------------------------------------
  Double_t r = 0.;
  const Float_t pi2 = 2. * TMath::Pi();
  for (Int_t i = 0; i < mN; i++) {
    Cluster* c = mClusters[i];
    Double_t x = c->getX(), y = c->getY();
    r += TMath::Sqrt(x * x + y * y);

    c->getXAlphaRefPlane(mXRef[i], mAlphaRef[i]);
    Float_t xyz[3];
    c->getGlobalXYZ(xyz);
    Float_t phi = TMath::ATan2(xyz[1], xyz[0]);
    if (phi < 0.)
      phi += pi2;
    else if (phi >= pi2)
      phi -= pi2;
    mPhi[i] = phi;
  }
  if (mN)
    mR = r / mN;
}

void CookedTracker::Layer::unloadClusters()
{
  //--------------------------------------------------------------------
  // Load clusters to this layer
  //--------------------------------------------------------------------
  // for (Int_t i=0; i<mN; i++) {delete mClusters[i]; mClusters[i]=0;}
  mN = 0;
}

Bool_t CookedTracker::Layer::insertCluster(Cluster* c)
{
  //--------------------------------------------------------------------
  // This function inserts a cluster to this layer in increasing
  // order of the cluster's fZ
  //--------------------------------------------------------------------
  if (mN >= kMaxClusterPerLayer) {
    ::Error("InsertCluster", "Too many clusters !\n");
    return kFALSE;
  }
  if (mN == 0)
    mClusters[0] = c;
  else {
    Int_t i = findClusterIndex(c->getZ());
    Int_t k = mN - i;
    memmove(mClusters + i + 1, mClusters + i, k * sizeof(Cluster*));
    mClusters[i] = c;
  }
  mN++;
  return kTRUE;
}

Int_t CookedTracker::Layer::findClusterIndex(Double_t z) const
{
  //--------------------------------------------------------------------
  // This function returns the index of the first
  // with its fZ >= "z".
  //--------------------------------------------------------------------
  if (mN == 0)
    return 0;

  Int_t b = 0;
  if (z <= mClusters[b]->getZ())
    return b;

  Int_t e = b + mN - 1;
  if (z > mClusters[e]->getZ())
    return e + 1;

  Int_t m = (b + e) / 2;
  for (; b < e; m = (b + e) / 2) {
    if (z > mClusters[m]->getZ())
      b = m + 1;
    else
      e = m;
  }
  return m;
}

void CookedTracker::Layer::selectClusters(Int_t& n, Int_t idx[], Float_t phi, Float_t dy, Float_t z, Float_t dz)
{
  //--------------------------------------------------------------------
  // This function selects clusters within the "road"
  //--------------------------------------------------------------------
  Float_t dphi = dy / mR;
  Float_t phiMin = phi - dphi;
  Float_t phiMax = phi + dphi;
  Float_t zMin = z - dz;
  Float_t zMax = z + dz;

  Int_t imin = findClusterIndex(zMin), imax = findClusterIndex(zMax);
  for (Int_t i = imin; i < imax; i++) {
    Float_t cphi = mPhi[i];
    if (cphi <= phiMin)
      continue;
    if (cphi > phiMax)
      continue;

    idx[n++] = i;
    if (n >= kMaxSelected)
      return;
  }
}

Bool_t CookedTracker::attachCluster(Int_t& volID, Int_t nl, Int_t ci, CookedTrack& t, const CookedTrack& o) const
{
  Layer& layer = sLayers[nl];
  Cluster* c = layer.getCluster(ci);

  Int_t vid = c->getVolumeId();

  if (vid != volID) {
    volID = vid;
    t = o;
    Double_t x = layer.getXRef(ci);
    Double_t alpha = layer.getAlphaRef(ci);
    if (!t.propagate(alpha, x, getBz()))
      return kFALSE;
  }

  Double_t chi2 = t.getPredictedChi2(c);
  if (chi2 > kmaxChi2PerCluster)
    return kFALSE;

  if (!t.update(c, chi2, (nl << 28) + ci))
    return kFALSE;

  Double_t xx0 = (nl > 2) ? 0.008 : 0.003; // Rough layer thickness
  Double_t x0 = 9.36;                      // Radiation length of Si [cm]
  Double_t rho = 2.33;                     // Density of Si [g/cm^3]
  t.correctForMeanMaterial(xx0, xx0 * x0 * rho, kTRUE);

  return kTRUE;
}
