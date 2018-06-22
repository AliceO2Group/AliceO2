// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CookedTracker.cxx
/// \brief Implementation of the "Cooked Matrix" ITS tracker
/// \author iouri.belikov@cern.ch

//-------------------------------------------------------------------------
//                     A stand-alone ITS tracker
//    The pattern recongintion based on the "cooked covariance" approach
//-------------------------------------------------------------------------
#include <chrono>
#include <future>

#include <TGeoGlobalMagField.h>
#include <TMath.h>

#include "FairLogger.h"

#include "CommonConstants/MathConstants.h"
#include "DetectorsBase/Propagator.h"
#include "Field/MagneticField.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "ITSReconstruction/CookedTracker.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::ITS;
using namespace o2::ITSMFT;
using namespace o2::constants::math;
using namespace o2::utils;
using o2::field::MagneticField;
using Label = o2::MCCompLabel;
using Point3Df = Point3D<float>;

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

CookedTracker::CookedTracker(Int_t n) : mNumOfThreads(n), mBz(0.)
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
  const Double_t klRadius[7] = { 2.34, 3.15, 3.93, 19.61, 24.55, 34.39, 39.34 }; // tdr6

  for (Int_t i = 0; i < kNLayers; i++)
    sLayers[i].setR(klRadius[i]);
}

//__________________________________________________________________________
Label CookedTracker::cookLabel(TrackITS& t, Float_t wrong) const
{
  //--------------------------------------------------------------------
  // This function "cooks" a track label.
  // A label<0 indicates that some of the clusters are wrongly assigned.
  //--------------------------------------------------------------------
  Int_t noc = t.getNumberOfClusters();
  std::array<Label, Cluster::maxLabels * CookedTracker::kNLayers> lb;
  std::array<Int_t, Cluster::maxLabels * CookedTracker::kNLayers> mx;

  int nLabels = 0;

  for (int i = noc; i--;) {
    Int_t index = t.getClusterIndex(i);
    const Cluster* c = getCluster(index);
    Int_t idx = c - mFirstCluster; // Index of this cluster in event
    auto labels = mClsLabels->getLabels(idx);

    for (auto lab : labels) { // check all labels of the cluster
      if (lab.isEmpty())
        break; // all following labels will be empty also
      // was this label already accounted for ?
      bool add = true;
      for (int j = nLabels; j--;) {
        if (lb[j] == lab) {
          add = false;
          mx[j]++; // just increment counter
          break;
        }
      }
      if (add) {
        lb[nLabels] = lab;
        mx[nLabels] = 1;
        nLabels++;
      }
    }
  }
  Label lab;
  Int_t maxL = 0; // find most encountered label
  for (int i = nLabels; i--;) {
    if (mx[i] > maxL) {
      maxL = mx[i];
      lab = lb[i];
    }
  }

  if ((1. - Float_t(maxL) / noc) > wrong) {
    // change the track ID to negative
    lab.set(-lab.getTrackID(), lab.getEventID(), lab.getSourceID());
  }
  // t.SetFakeRatio((1.- Float_t(maxL)/noc));
  return lab;
}

//__________________________________________________________________________
void CookedTracker::setExternalIndices(TrackITS& t) const
{
  //--------------------------------------------------------------------
  // Set the indices within the external cluster array.
  //--------------------------------------------------------------------
  Int_t noc = t.getNumberOfClusters();
  for (Int_t i = 0; i < noc; i++) {
    Int_t index = t.getClusterIndex(i);
    const Cluster* c = getCluster(index);
    Int_t idx = c - mFirstCluster; // Index of this cluster in event
    t.setExternalClusterIndex(i, idx);
  }
}

Double_t CookedTracker::getBz() const
{
  return mBz;
  /*
  MagneticField* fld = (MagneticField*)TGeoGlobalMagField::Instance()->GetField();
  if (!fld) {
    LOG(FATAL) << "Field is not loaded !" << FairLogger::endl;
    return Almost0;
  }
  Double_t bz = fld->solenoidField();
  return TMath::Sign(Almost0, bz) + bz;
  */
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

static TrackITS cookSeed(const Point3Df& r1, Point3Df& r2, const Point3Df& tr3, float rad2, float rad3, float_t alpha,
                         float_t bz)
// const  Float_t r1[4], const Float_t r2[4], const Float_t tr3[4], Double_t alpha, Double_t bz)
{
  //--------------------------------------------------------------------
  // This is the main cooking function.
  // Creates seed parameters out of provided clusters.
  //--------------------------------------------------------------------

  Double_t ca = TMath::Cos(alpha), sa = TMath::Sin(alpha);
  Double_t x1 = r1.X() * ca + r1.Y() * sa, y1 = -r1.X() * sa + r1.Y() * ca, z1 = r1.Z();
  Double_t x2 = r2.X() * ca + r2.Y() * sa, y2 = -r2.X() * sa + r2.Y() * ca, z2 = r2.Z();
  Double_t x3 = tr3.X(), y3 = tr3.Y(), z3 = tr3.Z();

  std::array<float, 5> par;
  par[0] = y3;
  par[1] = z3;
  Double_t crv = f1(x1, y1, x2, y2, x3, y3); // curvature
  Double_t x0 = f2(x1, y1, x2, y2, x3, y3);  // x-coordinate of the center
  Double_t tgl12 = f3(x1, y1, x2, y2, z1, z2);
  Double_t tgl23 = f3(x2, y2, x3, y3, z2, z3);

  Double_t sf = crv * (x3 - x0); // FIXME: sf must never be >= kAlmost1
  par[2] = sf;

  par[3] = 0.5 * (tgl12 + tgl23);
  par[4] = (TMath::Abs(bz) < Almost0) ? Almost0 : crv / (bz * B2C);

  std::array<float, 15> cov;
  /*
  for (Int_t i=0; i<15; i++) cov[i]=0.;
  cov[0] =kSigma2*10;
  cov[2] =kSigma2*10;
  cov[5] =0.007*0.007*10;   //FIXME all these lines
  cov[9] =0.007*0.007*10;
  cov[14]=0.1*0.1*10;
  */
  const Double_t dlt = 0.0005;
  Double_t fy = 1. / (rad2 - rad3);
  Double_t tz = fy;
  Double_t cy = (f1(x1, y1, x2, y2 + dlt, x3, y3) - crv) / dlt / bz / B2C;
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

  return TrackITS(x3, alpha, par, cov);
}

void CookedTracker::makeSeeds(std::vector<TrackITS>& seeds, Int_t first, Int_t last)
{
  //--------------------------------------------------------------------
  // This is the main pattern recongition function.
  // Creates seeds out of two clusters and another point.
  //--------------------------------------------------------------------
  const Double_t zv = getZ();

  Layer& layer1 = sLayers[kSeedingLayer1];
  Layer& layer2 = sLayers[kSeedingLayer2];
  Layer& layer3 = sLayers[kSeedingLayer3];

  const Double_t maxC = TMath::Abs(getBz() * B2C / kminPt);
  const Double_t kpWin = TMath::ASin(0.5 * maxC * layer1.getR()) - TMath::ASin(0.5 * maxC * layer2.getR());

  // Int_t nClusters1 = layer1.getNumberOfClusters();
  Int_t nClusters2 = layer2.getNumberOfClusters();
  Int_t nClusters3 = layer3.getNumberOfClusters();

  for (Int_t n1 = first; n1 < last; n1++) {
    const Cluster* c1 = layer1.getCluster(n1);
    //
    // Int_t lab=c1->getLabel(0);
    //
    Double_t z1 = c1->getZ();
    auto xyz1 = c1->getXYZGloRot(*mGeom);
    Double_t r1 = xyz1.rho(), phi1 = layer1.getClusterPhi(n1);

    Double_t zr2 = zv + layer2.getR() / r1 * (z1 - zv);
    Int_t start2 = layer2.findClusterIndex(zr2 - kzWin);

    for (Int_t n2 = start2; n2 < nClusters2; n2++) {
      const Cluster* c2 = layer2.getCluster(n2);
      //
      // if (c2->getLabel(0)!=lab) continue;
      //
      Double_t z2 = c2->getZ();
      if (z2 > (zr2 + kzWin))
        break; // check in Z

      Double_t phi2 = layer2.getClusterPhi(n2);
      if (TMath::Abs(phi2 - phi1) > kpWin)
        continue; // check in Phi

      auto xyz2 = c2->getXYZGloRot(*mGeom);
      Double_t r2 = xyz2.rho();
      Double_t crv = f1(xyz1.X(), xyz1.Y(), xyz2.X(), xyz2.Y(), getX(), getY());

      Double_t zr3 = z1 + (layer3.getR() - r1) / (r2 - r1) * (z2 - z1);
      Double_t dz = kzWin / 2;

      Int_t start3 = layer3.findClusterIndex(zr3 - dz);
      for (Int_t n3 = start3; n3 < nClusters3; n3++) {
        const Cluster* c3 = layer3.getCluster(n3);
        //
        // if (c3->getLabel(0)!=lab) continue;
        //
        Double_t z3 = c3->getZ();
        if (z3 > (zr3 + dz))
          break; // check in Z

        Double_t r3 = c3->getX();
        Double_t phir3 = phi1 + 0.5 * crv * (r3 - r1);
        Double_t phi3 = layer3.getClusterPhi(n3);
        if (TMath::Abs(phir3 - phi3) > kpWin / 100)
          continue; // check in Phi

        Point3Df txyz2 = c2->getXYZ(); // tracking coordinates
        //      txyz2.SetX(layer2.getXRef(n2));  // The clusters are already in the tracking frame

        auto xyz3 = c3->getXYZGloRot(*mGeom);

        TrackITS seed = cookSeed(xyz1, xyz3, txyz2, layer2.getR(), layer3.getR(), layer2.getAlphaRef(n2), getBz());

        float ip[2];
        seed.getImpactParams(getX(), getY(), getZ(), getBz(), ip);
        if (TMath::Abs(ip[0]) > kmaxDCAxy)
          continue;
        if (TMath::Abs(ip[1]) > kmaxDCAz)
          continue;
        {
          Double_t xx0 = 0.008; // Rough layer thickness
          Double_t radl = 9.36; // Radiation length of Si [cm]
          Double_t rho = 2.33;  // Density of Si [g/cm^3]
          if (!seed.correctForMaterial(xx0, xx0 * radl * rho, kTRUE))
            continue;
        }
        seed.setClusterIndex(kSeedingLayer1, n1);
        seed.setClusterIndex(kSeedingLayer3, n3);
        seed.setClusterIndex(kSeedingLayer2, n2);
        seeds.push_back(seed);
      }
    }
  }
  /*
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
  */
}

void CookedTracker::trackSeeds(std::vector<TrackITS>& seeds)
{
  //--------------------------------------------------------------------
  // Loop over a subset of track seeds
  //--------------------------------------------------------------------
  std::vector<bool> used[kSeedingLayer2];
  std::vector<Int_t> selec[kSeedingLayer2];
  for (Int_t l = kSeedingLayer2 - 1; l >= 0; l--) {
    Int_t n = sLayers[l].getNumberOfClusters();
    used[l].resize(n, false);
    selec[l].reserve(n / 100);
  }

  for (auto& track : seeds) {
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
      selec[l].clear();
      sLayers[l].selectClusters(selec[l], phi, kRoadY, z, kRoadZ);
      r1 = r2;
    }

    TrackITS best(track);

    Int_t volID = -1;
    Int_t ci = -1;
    TrackITS t3(track);
    for (auto& ci3 : selec[3]) {
      if (used[3][ci3])
        continue;
      if (!attachCluster(volID, 3, ci3, t3, track))
        continue;

      TrackITS t2(t3);
      for (auto& ci2 : selec[2]) {
        if (used[2][ci2])
          continue;
        if (!attachCluster(volID, 2, ci2, t2, t3))
          continue;

        TrackITS t1(t2);
        for (auto& ci1 : selec[1]) {
          if (used[1][ci1])
            continue;
          if (!attachCluster(volID, 1, ci1, t1, t2))
            continue;

          TrackITS t0(t1);
          for (auto& ci0 : selec[0]) {
            if (used[0][ci0])
              continue;
            if (!attachCluster(volID, 0, ci0, t0, t1))
              continue;
            if (t0.isBetter(best, kmaxChi2PerTrack)) {
              best = t0;
            }
            volID = -1;
          }
        }
      }
    }

    if (best.getNumberOfClusters() >= kminNumberOfClusters) {
      Int_t noc = best.getNumberOfClusters();
      for (Int_t ic = 3; ic < noc; ic++) {
        Int_t index = best.getClusterIndex(ic);
        Int_t l = (index & 0xf0000000) >> 28, c = (index & 0x0fffffff);
        used[l][c] = true;
      }
    }
    track = best;
  }
}

std::vector<TrackITS> CookedTracker::trackInThread(Int_t first, Int_t last)
{
  //--------------------------------------------------------------------
  // This function is passed to a tracking thread
  //--------------------------------------------------------------------
  std::vector<TrackITS> seeds;
  seeds.reserve(last - first + 1);

  for (auto& vtx : mVertices) {
    mX = vtx[0];
    mY = vtx[1];
    mZ = vtx[2];
    makeSeeds(seeds, first, last);
  }

  std::sort(seeds.begin(), seeds.end());

  trackSeeds(seeds);

  makeBackPropParam(seeds);

  return seeds;
}

void CookedTracker::process(const std::vector<Cluster>& clusters, std::vector<TrackITS>& tracks)
{
  //--------------------------------------------------------------------
  // This is the main tracking function
  //--------------------------------------------------------------------
  static int entry = 0;

  LOG(INFO) << FairLogger::endl;
  LOG(INFO) << "CookedTracker::process() entry " << entry++ << ", number of threads: " << mNumOfThreads
            << FairLogger::endl;

  int nClFrame = 0;
  Int_t numOfClustersLeft = clusters.size(); // total number of clusters
  if (numOfClustersLeft == 0) {
    LOG(WARNING) << "No clusters to process !" << FairLogger::endl;
    return;
  }

  auto start = std::chrono::system_clock::now();

  while (numOfClustersLeft > 0) {
    nClFrame = loadClusters(clusters);
    if (!nClFrame) {
      LOG(FATAL) << "Failed to select any cluster out of " << numOfClustersLeft << " check if cont/trig mode is correct"
                 << FairLogger::endl;
    }
    numOfClustersLeft -= nClFrame;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    LOG(INFO) << "Loading clusters: " << nClFrame << " in single frame " << mROFrame << " : " << diff.count() << " s"
              << FairLogger::endl;

    start = end;

    processFrame(tracks);
    unloadClusters();
    end = std::chrono::system_clock::now();
    diff = end - start;
    LOG(INFO) << "Processing time for single frame " << mROFrame << " : " << diff.count() << " s" << FairLogger::endl;

    start = end;
    if (mContinuousMode)
      mROFrame++; // expect incremented frame in following clusters
  }
}

void CookedTracker::processFrame(std::vector<TrackITS>& tracks)
{
  //--------------------------------------------------------------------
  // This is the main tracking function for single frame, it is assumed that only clusters
  // which may contribute to this frame is loaded
  //--------------------------------------------------------------------
  LOG(INFO) << "CookedTracker::process(), number of threads: " << mNumOfThreads << FairLogger::endl;

  std::vector<std::future<std::vector<TrackITS>>> futures(mNumOfThreads);
  std::vector<std::vector<TrackITS>> seedArray(mNumOfThreads);

  Int_t numOfClusters = sLayers[kSeedingLayer1].getNumberOfClusters();
  for (Int_t t = 0, first = 0; t < mNumOfThreads; t++) {
    Int_t rem = t < (numOfClusters % mNumOfThreads) ? 1 : 0;
    Int_t last = first + (numOfClusters / mNumOfThreads) + rem;
    futures[t] = std::async(std::launch::async, &CookedTracker::trackInThread, this, first, last);
    first = last;
  }
  Int_t nSeeds = 0, ngood = 0;
  for (Int_t t = 0; t < mNumOfThreads; t++) {
    seedArray[t] = futures[t].get();
    nSeeds += seedArray[t].size();
    for (auto& track : seedArray[t]) {
      if (track.getNumberOfClusters() < kminNumberOfClusters)
        continue;
      if (mTrkLabels) {
        Label label = cookLabel(track, 0.); // For comparison only
        if (label.getTrackID() >= 0)
          ngood++;
        Int_t idx = tracks.size();
        mTrkLabels->addElement(idx, label);
      }
      setExternalIndices(track);
      track.setROFrame(mROFrame);
      tracks.push_back(track);
    }
  }

  if (nSeeds) {
    LOG(INFO) << "CookedTracker::process(), good_tracks:/seeds: " << ngood << '/' << nSeeds << "-> "
              << Float_t(ngood) / nSeeds << FairLogger::endl;
  }
}

//____________________________________________________________
void CookedTracker::makeBackPropParam(std::vector<TrackITS>& seeds) const
{
  // refit in backward direction
  for (auto& track : seeds) {
    if (track.getNumberOfClusters() < kminNumberOfClusters)
      continue;
    makeBackPropParam(track);
  }
}

//____________________________________________________________
bool CookedTracker::makeBackPropParam(TrackITS& track) const
{
  // refit in backward direction
  auto backProp = track.getParamOut();
  backProp = track;
  backProp.resetCovariance();
  auto propagator = o2::Base::Propagator::Instance();

  Int_t noc = track.getNumberOfClusters();
  for (int ic = noc; ic--;) { // cluster indices are stored in inward direction
    Int_t index = track.getClusterIndex(ic);
    const Cluster* c = getCluster(index);
    float alpha = mGeom->getSensorRefAlpha(c->getSensorID());
    if (!backProp.rotate(alpha)) {
      return false;
    }
    if (!propagator->PropagateToXBxByBz(backProp, c->getX())) {
      return false;
    }
    if (!backProp.update(static_cast<const o2::BaseCluster<float>&>(*c))) {
      return false;
    }
  }
  track.getParamOut() = backProp;
  return true;
}

int CookedTracker::loadClusters(const std::vector<Cluster>& clusters)
{
  //--------------------------------------------------------------------
  // This function reads the ITSU clusters from the tree,
  // sort them, distribute over the internal tracker arrays, etc
  //--------------------------------------------------------------------
  Int_t numOfClusters = clusters.size();
  int nLoaded = 0;
  mFirstCluster = &clusters.front();
  
  if (mContinuousMode) { // check the ROFrame in cont. mode
    for (auto& c : clusters) {
      if (c.getROFrame() != mROFrame) {
        continue;
      }
      nLoaded++;
      Int_t layer = mGeom->getLayer(c.getSensorID());
      if (!sLayers[layer].insertCluster(&c)) {
        continue;
      }
    }
  } else { // do not check the ROFrame in triggered mode
    for (auto& c : clusters) {
      nLoaded++;
      Int_t layer = mGeom->getLayer(c.getSensorID());
      if (!sLayers[layer].insertCluster(&c)) {
        continue;
      }
    }
  }

  if (nLoaded) {
    std::vector<std::future<void>> fut;
    for (Int_t l = 0; l < kNLayers; l += mNumOfThreads) {
      for (Int_t t = 0; t < mNumOfThreads; t++) {
        if (l + t >= kNLayers)
          break;
        auto f = std::async(std::launch::async, &CookedTracker::Layer::init, sLayers + (l + t));
        fut.push_back(std::move(f));
      }
      for (Int_t t = 0; t < fut.size(); t++)
        fut[t].wait();
    }
  }
  return nLoaded;
}

void CookedTracker::unloadClusters()
{
  //--------------------------------------------------------------------
  // This function unloads ITSU clusters from the RAM
  //--------------------------------------------------------------------
  for (Int_t i = 0; i < kNLayers; i++)
    sLayers[i].unloadClusters();
}

const Cluster* CookedTracker::getCluster(Int_t index) const
{
  //--------------------------------------------------------------------
  //       Return pointer to a given cluster
  //--------------------------------------------------------------------
  Int_t l = (index & 0xf0000000) >> 28;
  Int_t c = (index & 0x0fffffff) >> 00;
  return sLayers[l].getCluster(c);
}

CookedTracker::Layer::Layer() : mR(0)
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
}

void CookedTracker::Layer::init()
{
  //--------------------------------------------------------------------
  // Sort clusters and cache their reference plane info in a thread
  //--------------------------------------------------------------------
  std::sort(std::begin(mClusters), std::end(mClusters),
            [](const Cluster* c1, const Cluster* c2) { return (c1->getZ() < c2->getZ()); });

  Double_t r = 0.;
  const Float_t pi2 = 2. * TMath::Pi();
  Int_t m = mClusters.size();
  for (Int_t i = 0; i < m; i++) {
    const Cluster* c = mClusters[i];
    // Float_t xRef, aRef;
    // mGeom->getSensorXAlphaRefPlane(c->getSensorID(),xRef, aRef);
    mAlphaRef.push_back(mGeom->getSensorRefAlpha(c->getSensorID()));
    auto xyz = c->getXYZGloRot(*mGeom);
    r += xyz.rho();
    Float_t phi = xyz.Phi();
    BringTo02Pi(phi);
    mPhi.push_back(phi);
    Int_t s = phi * kNSectors / pi2;
    mSectors[s].push_back(i);
  }

  if (m)
    mR = r / m;
}

void CookedTracker::Layer::unloadClusters()
{
  //--------------------------------------------------------------------
  // Unload clusters from this layer
  //--------------------------------------------------------------------
  mClusters.clear();
  mAlphaRef.clear();
  mPhi.clear();
  for (Int_t s = 0; s < kNSectors; s++)
    mSectors[s].clear();
}

Bool_t CookedTracker::Layer::insertCluster(const Cluster* c)
{
  //--------------------------------------------------------------------
  // This function inserts a cluster to this layer
  //--------------------------------------------------------------------
  mClusters.push_back(c);
  return kTRUE;
}

Int_t CookedTracker::Layer::findClusterIndex(Double_t z) const
{
  //--------------------------------------------------------------------
  // This function returns the index of the first cluster with its fZ >= "z".
  //--------------------------------------------------------------------
  auto found = std::upper_bound(std::begin(mClusters), std::end(mClusters), z,
                                [](Double_t zc, const Cluster* c) { return (zc < c->getZ()); });
  return found - std::begin(mClusters);
}

void CookedTracker::Layer::selectClusters(std::vector<Int_t>& selec, Float_t phi, Float_t dy, Float_t z, Float_t dz)
{
  //--------------------------------------------------------------------
  // This function selects clusters within the "road"
  //--------------------------------------------------------------------
  Float_t zMin = z - dz;
  Float_t zMax = z + dz;

  const Float_t pi2 = 2. * TMath::Pi();
  Float_t dphi = dy / mR;

  Float_t phiMin = phi - dphi;
  Float_t phiMax = phi + dphi;
  Float_t phiRange[2]{ phiMin, phiMax };

  Int_t n = 0;
  Int_t sector = -1;
  for (auto phiM : phiRange) {
    Int_t s = phiM * kNSectors / pi2;
    if (s < 0)
      s += kNSectors;
    else if (s >= kNSectors)
      s -= kNSectors;

    if (s == sector)
      break;
    sector = s;

    auto cmp = [this](Double_t zc, Int_t ic) { return (zc < mClusters[ic]->getZ()); };
    auto imin = std::upper_bound(std::begin(mSectors[s]), std::end(mSectors[s]), zMin, cmp);
    auto imax = std::upper_bound(imin, std::end(mSectors[s]), zMax, cmp);
    for (; imin != imax; imin++) {
      Int_t i = *imin;
      Float_t cphi = mPhi[i];
      if (cphi <= phiMin)
        continue;
      if (cphi > phiMax)
        continue;

      selec.push_back(i);
    }
  }
}

Bool_t CookedTracker::attachCluster(Int_t& volID, Int_t nl, Int_t ci, TrackITS& t, const TrackITS& o) const
{
  //--------------------------------------------------------------------
  // Try to attach a clusters with index ci to running track hypothesis
  //--------------------------------------------------------------------
  Layer& layer = sLayers[nl];
  const Cluster* c = layer.getCluster(ci);

  Int_t vid = c->getSensorID();

  if (vid != volID) {
    volID = vid;
    t = o;
    Double_t alpha = layer.getAlphaRef(ci);
    if (!t.propagate(alpha, c->getX(), getBz()))
      return kFALSE;
  }

  Double_t chi2 = t.getPredictedChi2(*c);

  if (chi2 > kmaxChi2PerCluster)
    return kFALSE;

  if (!t.update(*c, chi2, (nl << 28) + ci))
    return kFALSE;

  Double_t xx0 = (nl > 2) ? 0.008 : 0.003; // Rough layer thickness
  Double_t x0 = 9.36;                      // Radiation length of Si [cm]
  Double_t rho = 2.33;                     // Density of Si [g/cm^3]
  t.correctForMaterial(xx0, xx0 * x0 * rho, kTRUE);
  return kTRUE;
}

void CookedTracker::setGeometry(o2::ITS::GeometryTGeo* geom)
{
  /// attach geometry interface
  mGeom = geom;
  for (Int_t i = 0; i < kNLayers; i++)
    sLayers[i].setGeometry(geom);
}
