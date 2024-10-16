// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMOfflineFitter.cxx
/// \author Sergey Gorbunov

#if (defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_GPUCODE))

#include "GPUTPCGMOfflineFitter.h"

#include "GPUCommonMath.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "AliHLTTPCGeometry.h"
#include <cmath>
#include "AliTracker.h"
#include "AliMagF.h"
#include "AliExternalTrackParam.h"
#include "AliTPCtracker.h"
#include "AliTPCParam.h"
#include "AliTPCseed.h"
#include "AliTPCclusterMI.h"
#include "AliTPCcalibDB.h"
#include "AliTPCParamSR.h"
#include "GPUTPCGMPropagator.h"
#include "AliTPCReconstructor.h"
#include "AliHLTTPCClusterTransformation.h"

#define DOUBLE 1

GPUTPCGMOfflineFitter::GPUTPCGMOfflineFitter() : fCAParam() {}

GPUTPCGMOfflineFitter::~GPUTPCGMOfflineFitter() {}

void GPUTPCGMOfflineFitter::Initialize(const GPUParam& hltParam, long TimeStamp, bool isMC)
{
  //

  AliHLTTPCClusterTransformation hltTransform;
  hltTransform.Init(0., TimeStamp, isMC, 1);

  // initialisation of AliTPCtracker as it is done in AliTPCReconstructor.cxx

  AliTPCcalibDB* calib = AliTPCcalibDB::Instance();
  const AliMagF* field = (AliMagF*)TGeoGlobalMagField::Instance()->GetField();
  calib->SetExBField(field);

  AliTPCParam* param = AliTPCcalibDB::Instance()->GetParameters();
  if (!param) {
    AliWarning("Loading default TPC parameters !");
    param = new AliTPCParamSR;
  }
  param->ReadGeoMatrices();

  AliTPCReconstructor* tpcRec = new AliTPCReconstructor();
  tpcRec->SetRecoParam(AliTPCcalibDB::Instance()->GetTransform()->GetCurrentRecoParam());

  //(this)->~AliTPCtracker();   //call the destructor explicitly
  // new (this) AliTPCtracker(param); // call the constructor

  AliTPCtracker::fSectors = AliTPCtracker::fInnerSec;
  // AliTPCReconstructor::ParseOptions(tracker);  : not important, it only set useHLTClusters flag

  fCAParam = hltParam;
}

void GPUTPCGMOfflineFitter::RefitTrack(GPUTPCGMMergedTrack& track, const GPUTPCGMPolynomialField* field, GPUTPCGMMergedTrackHit* clusters)
{
  // copy of HLT RefitTrack() with calling of the offline fit utilities

  if (!track.OK()) {
    return;
  }

  int32_t nTrackHits = track.NClusters();
  cout << "call FitOffline .. " << endl;
  bool ok = FitOffline(field, track, clusters + track.FirstClusterRef(), nTrackHits);
  cout << ".. end of call FitOffline " << endl;

  GPUTPCGMTrackParam t = track.Param();
  float Alpha = track.Alpha();

  if (fabsf(t.QPt()) < 1.e-4) {
    t.QPt() = 1.e-4;
  }

  track.SetOK(ok);
  track.SetNClustersFitted(nTrackHits);
  track.Param() = t;
  track.Alpha() = Alpha;

  {
    int32_t ind = track.FirstClusterRef();
    float alphaa = fCAParam.Alpha(clusters[ind].slice);
    float xx = clusters[ind].fX;
    float yy = clusters[ind].fY;
    float zz = clusters[ind].fZ - track.Param().GetZOffset();
    float sinA = CAMath::Sin(alphaa - track.Alpha());
    float cosA = CAMath::Cos(alphaa - track.Alpha());
    track.SetLastX(xx * cosA - yy * sinA);
    track.SetLastY(xx * sinA + yy * cosA);
    track.SetLastZ(zz);
  }
}

int32_t GPUTPCGMOfflineFitter::CreateTPCclusterMI(const GPUTPCGMMergedTrackHit& h, AliTPCclusterMI& c)
{
  // Create AliTPCclusterMI for the HLT hit

  AliTPCclusterMI tmp; // everything is set to 0 by constructor
  c = tmp;

  // add the information we have

  Int_t sector, row;
  AliHLTTPCGeometry::Slice2Sector(h.slice, h.row, sector, row);
  c.SetDetector(sector);
  c.SetRow(row); // ?? is it right row numbering for the TPC tracker ??
  c.SetX(h.fX);
  c.SetY(h.fY);
  c.SetZ(h.fZ);
  int32_t index = (((sector << 8) + row) << 16) + 0;
  return index;
}

bool GPUTPCGMOfflineFitter::FitOffline(const GPUTPCGMPolynomialField* field, GPUTPCGMMergedTrack& gmtrack, GPUTPCGMMergedTrackHit* clusters, int32_t& N)
{
  const float maxSinPhi = GPUCA_MAX_SIN_PHI;

  int32_t maxN = N;
  float covYYUpd = 0.;
  float lastUpdateX = -1.;

  const bool rejectChi2ThisRound = 0;
  const bool markNonFittedClusters = 0;
  const double kDeg2Rad = 3.14159265358979323846 / 180.;
  const float maxSinForUpdate = CAMath::Sin(70. * kDeg2Rad);

  bool ok = 1;

  AliTPCtracker::SetIteration(2);

  AliTPCseed seed;
  gmtrack.Param().GetExtParam(seed, gmtrack.Alpha());

  AliTPCtracker::AddCovariance(&seed);

  N = 0;
  lastUpdateX = -1;

  // find last leg
  int32_t ihitStart = 0;
  for (int32_t ihit = 0; ihit < maxN; ihit++) {
    if (clusters[ihit].leg != clusters[ihitStart].leg) {
      ihitStart = ihit;
    }
  }

  for (int32_t ihit = ihitStart; ihit < maxN; ihit++) {
    if (clusters[ihit].fState < 0) {
      continue; // hit is excluded from fit
    }
    float xx = clusters[ihit].fX;
    float yy = clusters[ihit].fY;
    float zz = clusters[ihit].fZ;

    if (DOUBLE && ihit + 1 >= 0 && ihit + 1 < maxN && clusters[ihit].row == clusters[ihit + 1].row) {
      float count = 1.;
      do {
        if (clusters[ihit].slice != clusters[ihit + 1].slice || clusters[ihit].leg != clusters[ihit + 1].leg || fabsf(clusters[ihit].fY - clusters[ihit + 1].fY) > 4. || fabsf(clusters[ihit].fZ - clusters[ihit + 1].fZ) > 4.) {
          break;
        }
        ihit += 1;
        xx += clusters[ihit].fX;
        yy += clusters[ihit].fY;
        zz += clusters[ihit].fZ;
        count += 1.;
      } while (ihit + 1 >= 0 && ihit + 1 < maxN && clusters[ihit].row == clusters[ihit + 1].row);
      xx /= count;
      yy /= count;
      zz /= count;
    }

    // Create AliTPCclusterMI for the hit

    AliTPCclusterMI cluster;
    Int_t tpcindex = CreateTPCclusterMI(clusters[ihit], cluster);
    if (tpcindex < 0) {
      continue;
    }
    Double_t sy2 = 0, sz2 = 0;
    AliTPCtracker::ErrY2Z2(&seed, &cluster, sy2, sz2);
    cluster.SetSigmaY2(sy2);
    cluster.SetSigmaZ2(sz2);
    cluster.SetQ(10);
    cluster.SetMax(10);

    Int_t iRow = clusters[ihit].row;

    if (iRow < AliHLTTPCGeometry::GetNRowLow()) {
      AliTPCtracker::fSectors = AliTPCtracker::fInnerSec;
    } else {
      AliTPCtracker::fSectors = AliTPCtracker::fOuterSec;
    }

    seed.SetClusterIndex2(iRow, tpcindex);
    seed.SetClusterPointer(iRow, &cluster);
    seed.SetCurrentClusterIndex1(tpcindex);

    int32_t retVal;
    float threshold = 3. + (lastUpdateX >= 0 ? (fabsf(seed.GetX() - lastUpdateX) / 2) : 0.);
    if (N > 2 && (fabsf(yy - seed.GetY()) > threshold || fabsf(zz - seed.GetZ()) > threshold)) {
      retVal = 2;
    } else {
      Int_t err = !(AliTPCtracker::FollowToNext(seed, iRow));

      const int32_t err2 = N > 0 && CAMath::Abs(seed.GetSnp()) >= maxSinForUpdate;
      if (err || err2) {
        if (markNonFittedClusters) {
          if (N > 0 && (fabsf(yy - seed.GetY()) > 3 || fabsf(zz - seed.GetZ()) > 3)) {
            clusters[ihit].fState = -2;
          } else if (err && err >= -3) {
            clusters[ihit].fState = -1;
          }
        }
        continue;
      }

      // retVal = prop.Update( yy, zz, clusters[ihit].row, param, rejectChi2ThisRound);
      retVal = 0;
    }

    if (retVal == 0) // track is updated
    {
      lastUpdateX = seed.GetX();
      covYYUpd = seed.GetCovariance()[0];
      ihitStart = ihit;
      N++;
    } else if (retVal == 2) { // cluster far away form the track
      if (markNonFittedClusters) {
        clusters[ihit].fState = -2;
      }
    } else {
      break; // bad chi2 for the whole track, stop the fit
    }
  } // end loop over clusters

  GPUTPCGMTrackParam t;
  t.SetExtParam(seed);

  float Alpha = seed.GetAlpha();

  t.ConstrainSinPhi();

  bool ok1 = N >= GPUCA_TRACKLET_SELECTOR_MIN_HITS_B5(t.GetQPt()) && t.CheckNumericalQuality(covYYUpd);
  if (!ok1) {
    return (false);
  }

  //   const float kDeg2Rad = 3.1415926535897 / 180.f;
  const float kSectAngle = 2 * 3.1415926535897 / 18.f;

  if (fCAParam.GetTrackReferenceX() <= 500) {
    GPUTPCGMPropagator prop;
    prop.SetMaterialTPC();
    prop.SetPolynomialField(field);
    prop.SetMaxSinPhi(maxSinPhi);
    prop.SetToyMCEventsFlag(fCAParam.ToyMCEventsFlag());

    for (int32_t k = 0; k < 3; k++) // max 3 attempts
    {
      int32_t err = prop.PropagateToXAlpha(fCAParam.GetTrackReferenceX(), Alpha, 0);
      t.ConstrainSinPhi();
      if (fabsf(t.GetY()) <= t.GetX() * tan(kSectAngle / 2.f)) {
        break;
      }
      float dAngle = floor(atan2(t.GetY(), t.GetX()) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
      Alpha += dAngle;
      if (err || k == 2) {
        t.Rotate(dAngle);
        break;
      }
    }
  } else if (fabsf(t.GetY()) > t.GetX() * tan(kSectAngle / 2.f)) {
    float dAngle = floor(atan2(t.GetY(), t.GetX()) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
    t.Rotate(dAngle);
    Alpha += dAngle;
  }
  if (Alpha > 3.1415926535897) {
    Alpha -= 2 * 3.1415926535897;
  } else if (Alpha <= -3.1415926535897) {
    Alpha += 2 * 3.1415926535897;
  }

  gmtrack.Param() = t;
  gmtrack.Alpha() = Alpha;

  return (ok);
}

#endif
