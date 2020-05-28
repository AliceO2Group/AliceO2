// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMTrackParam.cxx
/// \author David Rohr, Sergey Gorbunov

#define GPUCA_CADEBUG 0
#define DEBUG_SINGLE_TRACK -1

#include "GPUTPCDef.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMPhysicalTrackModel.h"
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPolynomialField.h"
#include "GPUTPCGMMerger.h"
#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUdEdx.h"
#include "GPUParam.h"
#include "GPUTPCClusterErrorStat.h"
#include "GPUO2DataTypes.h"
#include "GPUConstantMem.h"
#include "TPCFastTransform.h"
#include "GPUTPCConvertImpl.h"
#include "GPUTPCGMMergerTypes.h"

#ifdef GPUCA_ALIROOT_LIB
#include "AliExternalTrackParam.h"
#endif

#ifdef GPUCA_CADEBUG_ENABLED
#include "../utils/qconfig.h"
#include "GPUChainTracking.h"
#include "AliHLTTPCClusterMCData.h"
#endif

#ifndef __OPENCL__
#include <cmath>
#include <cstdlib>
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

static constexpr float kRho = 1.025e-3f;  // 0.9e-3;
static constexpr float kRadLen = 29.532f; // 28.94;
static constexpr float kDeg2Rad = M_PI / 180.f;
static constexpr float kSectAngle = 2 * M_PI / 18.f;

GPUd() bool GPUTPCGMTrackParam::Fit(const GPUTPCGMMerger* GPUrestrict() merger, int iTrk, GPUTPCGMMergedTrackHit* GPUrestrict() clusters, int& GPUrestrict() N, int& GPUrestrict() NTolerated, float& GPUrestrict() Alpha, int attempt, float maxSinPhi, GPUTPCOuterParam* GPUrestrict() outerParam, GPUdEdxInfo* GPUrestrict() dEdxOut)
{
  const GPUParam& GPUrestrict() param = merger->Param();

  GPUTPCClusterErrorStat errorStat(N);

  GPUdEdx dEdx;
  GPUTPCGMPropagator prop;
  GPUTPCGMMergerTypes::InterpolationErrors interpolation;
  prop.SetMaterial(kRadLen, kRho);
  prop.SetPolynomialField(&merger->Param().polynomialField);
  prop.SetMaxSinPhi(maxSinPhi);
  prop.SetToyMCEventsFlag(param.ToyMCEventsFlag);
  prop.SetMatLUT(merger->GetConstantMem()->calibObjects.matLUT);
  if ((clusters[0].slice < 18) == (clusters[N - 1].slice < 18)) {
    ShiftZ2(clusters, merger, N);
  }
  if (param.rec.mergerInterpolateErrors) {
    for (int i = 0; i < N; i++) {
      interpolation.hit[i].errorY = -1;
    }
  }

  int nWays = param.rec.NWays;
  int maxN = N;
  int ihitStart = 0;
  float covYYUpd = 0.f;
  float lastUpdateX = -1.f;
  unsigned char lastRow = 255;
  unsigned char lastSlice = 255;

  for (int iWay = 0; iWay < nWays; iWay++) {
    int nMissed = 0, nMissed2 = 0;
    if (iWay && param.rec.NWaysOuter && iWay == nWays - 1 && outerParam) {
      for (int i = 0; i < 5; i++) {
        outerParam->P[i] = mP[i];
      }
      for (int i = 0; i < 15; i++) {
        outerParam->C[i] = mC[i];
      }
      outerParam->X = mX;
      outerParam->alpha = prop.GetAlpha();
    }

    int resetT0 = initResetT0();
    const bool refit = (nWays == 1 || iWay >= 1);
    const float maxSinForUpdate = CAMath::Sin(70.f * kDeg2Rad);

    ResetCovariance();
    prop.SetSeedingErrors(!(refit && attempt == 0));
    prop.SetFitInProjections(param.rec.fitInProjections == -1 ? (iWay != 0) : param.rec.fitInProjections);
    prop.SetPropagateBzOnly(param.rec.fitPropagateBzOnly > iWay);
    prop.SetTrack(this, iWay ? prop.GetAlpha() : Alpha);
    ConstrainSinPhi(prop.GetFitInProjections() ? 0.95f : GPUCA_MAX_SIN_PHI_LOW);
    CADEBUG(printf("Fitting track %d way %d (sector %d, alpha %f)\n", iTrk, iWay, (int)(prop.GetAlpha() / kSectAngle + 0.5) + (mP[1] < 0 ? 18 : 0), prop.GetAlpha()));

    N = 0;
    lastUpdateX = -1;
    const bool inFlyDirection = iWay & 1;
    unsigned char lastLeg = clusters[ihitStart].leg;
    const int wayDirection = (iWay & 1) ? -1 : 1;

    bool noFollowCircle = false, noFollowCircle2 = false;
    int goodRows = 0;
    for (int ihit = ihitStart; ihit >= 0 && ihit < maxN; ihit += wayDirection) {
      const bool crossCE = lastSlice != 255 && ((lastSlice < 18) ^ (clusters[ihit].slice < 18));
      if (crossCE) {
        lastSlice = clusters[ihit].slice;
        noFollowCircle2 = true;
        if (mC[2] < 0.5f) {
          mC[2] = 0.5f;
        }
      }

      unsigned char clusterState = clusters[ihit].state;
      const float clAlpha = param.Alpha(clusters[ihit].slice);
      if ((param.rec.RejectMode > 0 && nMissed >= param.rec.RejectMode) || nMissed2 >= GPUCA_MERGER_MAXN_MISSED_HARD || clusters[ihit].state & GPUTPCGMMergedTrackHit::flagReject) {
        CADEBUG(printf("\tSkipping hit, %d hits rejected, flag %X\n", nMissed, (int)clusters[ihit].state));
        if (iWay + 2 >= nWays && !(clusters[ihit].state & GPUTPCGMMergedTrackHit::flagReject)) {
          clusters[ihit].state |= GPUTPCGMMergedTrackHit::flagRejectErr;
        }
        continue;
      }

      const bool allowModification = refit && (iWay == 0 || (((nWays - iWay) & 1) ? (ihit >= CAMath::Min(maxN / 2, 30)) : (ihit <= CAMath::Max(maxN / 2, maxN - 30))));
      int ihitMergeFirst = ihit;
      prop.SetStatErrorCurCluster(&clusters[ihit]);

      float xx, yy, zz;
      if (merger->Param().earlyTpcTransform) {
        const float zOffset = (clusters[ihit].slice < 18) == (clusters[0].slice < 18) ? mTZOffset : -mTZOffset;
        xx = clusters[ihit].x;
        yy = clusters[ihit].y;
        zz = clusters[ihit].z - zOffset;
      } else {
        const ClusterNative& GPUrestrict() cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
        merger->GetConstantMem()->calibObjects.fastTransform->Transform(clusters[ihit].slice, clusters[ihit].row, cl.getPad(), cl.getTime(), xx, yy, zz, mTZOffset);
      }
      // clang-format off
      CADEBUG(printf("\tHit %3d/%3d Row %3d: Cluster Alpha %8.3f %3d, X %8.3f - Y %8.3f, Z %8.3f (Missed %d)", ihit, maxN, (int)clusters[ihit].row, clAlpha, (int)clusters[ihit].slice, xx, yy, zz, nMissed));
      CADEBUG(if ((unsigned int)merger->GetTrackingChain()->mIOPtrs.nMCLabelsTPC > clusters[ihit].num))
      CADEBUG({printf(" MC:"); for (int i = 0; i < 3; i++) {int mcId = merger->GetTrackingChain()->mIOPtrs.mcLabelsTPC[clusters[ihit].num].fClusterID[i].fMCID; if (mcId >= 0) printf(" %d", mcId); } } printf("\n"));
      // clang-format on
      if (MergeDoubleRowClusters(ihit, wayDirection, clusters, merger, prop, xx, yy, zz, maxN, clAlpha, clusterState, allowModification) == -1) {
        nMissed++;
        nMissed2++;
        continue;
      }

      bool changeDirection = (clusters[ihit].leg - lastLeg) & 1;
      // clang-format off
      CADEBUG(if (changeDirection) printf("\t\tChange direction\n"));
      CADEBUG(printf("\tLeg %3d%14sTrack   Alpha %8.3f %s, X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", (int)clusters[ihit].leg, "", prop.GetAlpha(), (CAMath::Abs(prop.GetAlpha() - clAlpha) < 0.01 ? "   " : " R!"), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10]));
      // clang-format on
      if (allowModification && changeDirection && !noFollowCircle && !noFollowCircle2) {
        bool tryFollow = lastRow != 255;
        if (tryFollow) {
          const GPUTPCGMTrackParam backup = *this;
          const float backupAlpha = prop.GetAlpha();
          if (FollowCircle(merger, prop, lastSlice, lastRow, iTrk, clusters[ihit].leg == clusters[maxN - 1].leg, clAlpha, xx, yy, clusters[ihit].slice, clusters[ihit].row, inFlyDirection)) {
            CADEBUG(printf("Error during follow circle, resetting track!\n"));
            *this = backup;
            prop.SetTrack(this, backupAlpha);
            noFollowCircle = true;
            tryFollow = false;
          }
        }
        if (tryFollow) {
          MirrorTo(prop, yy, zz, inFlyDirection, param, clusters[ihit].row, clusterState, false);
          lastUpdateX = mX;
          lastLeg = clusters[ihit].leg;
          lastSlice = clusters[ihit].slice;
          lastRow = 255;
          N++;
          resetT0 = initResetT0();
          // clang-format off
          CADEBUG(printf("\n"));
          CADEBUG(printf("\t%21sMirror  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10]));
          // clang-format on
          continue;
        }
      } else if (allowModification && lastRow != 255 && CAMath::Abs(clusters[ihit].row - lastRow) > 1) {
        if (dEdxOut && iWay == nWays - 1 && clusters[ihit].row == lastRow - 2 && clusters[ihit].leg == clusters[maxN - 1].leg) {
          dEdx.fillSubThreshold(lastRow - 1, param);
        }
        AttachClustersPropagate(merger, clusters[ihit].slice, lastRow, clusters[ihit].row, iTrk, clusters[ihit].leg == clusters[maxN - 1].leg, prop, inFlyDirection);
      }

      int err = prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection);
      // clang-format off
      CADEBUG(if (!CheckCov()){printf("INVALID COV AFTER PROPAGATE!!!\n");});
      // clang-format on
      if (err == -2) // Rotation failed, try to bring to new x with old alpha first, rotate, and then propagate to x, alpha
      {
        CADEBUG(printf("REROTATE\n"));
        if (prop.PropagateToXAlpha(xx, prop.GetAlpha(), inFlyDirection) == 0) {
          err = prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection);
        }
      }
      if (lastRow == 255 || CAMath::Abs((int)lastRow - (int)clusters[ihit].row) > 5 || lastSlice != clusters[ihit].slice || (param.rec.RejectMode < 0 && -nMissed <= param.rec.RejectMode)) {
        goodRows = 0;
      } else {
        goodRows++;
      }
      if (err == 0) {
        lastRow = clusters[ihit].row;
        lastSlice = clusters[ihit].slice;
      }
      // clang-format off
      CADEBUG(printf("\t%21sPropaga Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f)   ---   Res %8.3f %8.3f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   Err %d", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), mP[0] - yy, mP[1] - zz, sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10], err));
      // clang-format on

      if (err == 0 && changeDirection) {
        const float mirrordY = prop.GetMirroredYTrack();
        CADEBUG(printf(" -- MiroredY: %f --> %f", mP[0], mirrordY));
        if (CAMath::Abs(yy - mP[0]) > CAMath::Abs(yy - mirrordY)) {
          CADEBUG(printf(" - Mirroring!!!"));
          if (allowModification) {
            AttachClustersMirror(merger, clusters[ihit].slice, clusters[ihit].row, iTrk, yy, prop); // TODO: Never true, will always call FollowCircle above, really???
          }
          MirrorTo(prop, yy, zz, inFlyDirection, param, clusters[ihit].row, clusterState, true);
          noFollowCircle = false;

          lastUpdateX = mX;
          lastLeg = clusters[ihit].leg;
          lastRow = 255;
          N++;
          resetT0 = initResetT0();
          // clang-format off
          CADEBUG(printf("\n"));
          CADEBUG(printf("\t%21sMirror  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10]));
          // clang-format on
          continue;
        }
      }

      if (allowModification) {
        AttachClusters(merger, clusters[ihit].slice, clusters[ihit].row, iTrk, clusters[ihit].leg == clusters[maxN - 1].leg, prop);
      }

      const int err2 = mNDF > 0 && CAMath::Abs(prop.GetSinPhi0()) >= maxSinForUpdate;
      if (err || err2) {
        if (mC[0] > GPUCA_MERGER_COV_LIMIT || mC[2] > GPUCA_MERGER_COV_LIMIT) {
          break;
        }
        MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagNotFit);
        nMissed2++;
        NTolerated++;
        CADEBUG(printf(" --- break (%d, %d)\n", err, err2));
        continue;
      }
      CADEBUG(printf("\n"));
      errorStat.Fill(xx, yy, zz, prop.GetAlpha(), mX, mP, mC, ihit, iWay);

      int retVal;
      float threshold = 3.f + (lastUpdateX >= 0 ? (CAMath::Abs(mX - lastUpdateX) / 2) : 0.f);
      if (mNDF > 5 && (CAMath::Abs(yy - mP[0]) > threshold || CAMath::Abs(zz - mP[1]) > threshold)) {
        retVal = 2;
      } else {
        char rejectChi2 = attempt ? 0 : ((param.rec.mergerInterpolateErrors && CAMath::Abs(ihit - ihitMergeFirst) <= 1) ? (refit ? (2 + ((nWays - iWay) & 1)) : 0) : (allowModification && goodRows > 5));
        retVal = prop.Update(yy, zz, clusters[ihit].row, param, clusterState, rejectChi2, &interpolation.hit[ihit], refit);
      }
      // clang-format off
      CADEBUG(if (!CheckCov()) GPUError("INVALID COV AFTER UPDATE!!!"));
      CADEBUG(printf("\t%21sFit     Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f), DzDs %5.2f %16s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   Err %d\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), mP[3], "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10], retVal));
      // clang-format on

      if (retVal == 0) // track is updated
      {
        noFollowCircle2 = false;
        lastUpdateX = mX;
        covYYUpd = mC[0];
        nMissed = nMissed2 = 0;
        UnmarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagNotFit);
        N++;
        ihitStart = ihit;
        float dy = mP[0] - prop.Model().Y();
        float dz = mP[1] - prop.Model().Z();
        if (CAMath::Abs(mP[4]) > 10 && --resetT0 <= 0 && CAMath::Abs(mP[2]) < 0.15f && dy * dy + dz * dz > 1) {
          CADEBUG(printf("Reinit linearization\n"));
          prop.SetTrack(this, prop.GetAlpha());
        }
        if (dEdxOut && iWay == nWays - 1 && clusters[ihit].leg == clusters[maxN - 1].leg) {
          if (merger->GetConstantMem()->ioPtrs.clustersNative == nullptr) {
            dEdx.fillCluster(clusters[ihit].amp, 0, clusters[ihit].row, mP[2], mP[3], param, merger->GetConstantMem()->calibObjects.dEdxSplines, zz);
          } else {
            const ClusterNative& cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
            dEdx.fillCluster(cl.qTot, cl.qMax, clusters[ihit].row, mP[2], mP[3], param, merger->GetConstantMem()->calibObjects.dEdxSplines, zz);
          }
        }
      } else if (retVal == 2) { // cluster far away form the track
        if (allowModification) {
          MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagRejectDistance);
        }
        nMissed++;
        nMissed2++;
      } else {
        break; // bad chi2 for the whole track, stop the fit
      }
    }
    if (((nWays - iWay) & 1) && (clusters[0].slice < 18) == (clusters[maxN - 1].slice < 18)) {
      ShiftZ2(clusters, merger, N);
    }
  }
  ConstrainSinPhi();

  bool ok = N + NTolerated >= GPUCA_TRACKLET_SELECTOR_MIN_HITS(mP[4]) && CheckNumericalQuality(covYYUpd);
  if (!ok) {
    return (false);
  }

  if (dEdxOut) {
    dEdx.computedEdx(*dEdxOut, param);
  }
  Alpha = prop.GetAlpha();
  if (param.rec.TrackReferenceX <= 500) {
    for (int k = 0; k < 3; k++) // max 3 attempts
    {
      int err = prop.PropagateToXAlpha(param.rec.TrackReferenceX, Alpha, 0);
      ConstrainSinPhi();
      if (CAMath::Abs(mP[0]) <= mX * CAMath::Tan(kSectAngle / 2.f)) {
        break;
      }
      float dAngle = floor(CAMath::ATan2(mP[0], mX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
      Alpha += dAngle;
      if (err || k == 2) {
        Rotate(dAngle);
        ConstrainSinPhi();
        break;
      }
    }
  } else if (CAMath::Abs(mP[0]) > mX * CAMath::Tan(kSectAngle / 2.f)) {
    float dAngle = floor(CAMath::ATan2(mP[0], mX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
    Rotate(dAngle);
    ConstrainSinPhi();
    Alpha += dAngle;
  }
  if (Alpha > M_PI) {
    Alpha -= 2 * M_PI;
  } else if (Alpha <= -M_PI) {
    Alpha += 2 * M_PI;
  }

  return (ok);
}

GPUd() void GPUTPCGMTrackParam::MirrorTo(GPUTPCGMPropagator& GPUrestrict() prop, float toY, float toZ, bool inFlyDirection, const GPUParam& param, unsigned char row, unsigned char clusterState, bool mirrorParameters)
{
  if (mirrorParameters) {
    prop.Mirror(inFlyDirection);
  }
  float err2Y, err2Z;
  prop.GetErr2(err2Y, err2Z, param, toZ, row, clusterState);
  prop.Model().Y() = mP[0] = toY;
  prop.Model().Z() = mP[1] = toZ;
  if (mC[0] < err2Y) {
    mC[0] = err2Y;
  }
  if (mC[2] < err2Z) {
    mC[2] = err2Z;
  }
  if (CAMath::Abs(mC[5]) < 0.1f) {
    mC[5] = mC[5] > 0 ? 0.1f : -0.1f;
  }
  if (mC[9] < 1.f) {
    mC[9] = 1.f;
  }
  mC[1] = mC[4] = mC[6] = mC[8] = mC[11] = mC[13] = 0;
  prop.SetTrack(this, prop.GetAlpha());
  mNDF = -3;
  mChi2 = 0;
}

GPUd() int GPUTPCGMTrackParam::MergeDoubleRowClusters(int& ihit, int wayDirection, GPUTPCGMMergedTrackHit* GPUrestrict() clusters, const GPUTPCGMMerger* GPUrestrict() merger, GPUTPCGMPropagator& GPUrestrict() prop, float& GPUrestrict() xx, float& GPUrestrict() yy, float& GPUrestrict() zz, int maxN, float clAlpha, unsigned char& GPUrestrict() clusterState, bool rejectChi2)
{
  if (ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].row == clusters[ihit + wayDirection].row && clusters[ihit].slice == clusters[ihit + wayDirection].slice && clusters[ihit].leg == clusters[ihit + wayDirection].leg) {
    float maxDistY, maxDistZ;
    prop.GetErr2(maxDistY, maxDistZ, merger->Param(), zz, clusters[ihit].row, 0);
    maxDistY = (maxDistY + mC[0]) * 20.f;
    maxDistZ = (maxDistZ + mC[2]) * 20.f;
    int noReject = 0; // Cannot reject if simple estimation of y/z fails (extremely unlike case)
    if (CAMath::Abs(clAlpha - prop.GetAlpha()) > 1.e-4f) {
      noReject = prop.RotateToAlpha(clAlpha);
    }
    float projY = 0, projZ = 0;
    if (noReject == 0) {
      noReject |= prop.GetPropagatedYZ(xx, projY, projZ);
    }
    float count = 0.f;
    xx = yy = zz = 0.f;
    clusterState = 0;
    while (true) {
      float clx, cly, clz, clamp;
      if (merger->Param().earlyTpcTransform) {
        const float zOffset = (clusters[ihit].slice < 18) == (clusters[0].slice < 18) ? mTZOffset : -mTZOffset;
        clx = clusters[ihit].x;
        cly = clusters[ihit].y;
        clz = clusters[ihit].z - zOffset;
        clamp = clusters[ihit].amp;
      } else {
        const ClusterNative& GPUrestrict() cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
        clamp = cl.qTot;
        merger->GetConstantMem()->calibObjects.fastTransform->Transform(clusters[ihit].slice, clusters[ihit].row, cl.getPad(), cl.getTime(), clx, cly, clz, mTZOffset);
      }
      float dy = cly - projY;
      float dz = clz - projZ;
      if (noReject == 0 && (dy * dy > maxDistY || dz * dz > maxDistZ)) {
        CADEBUG(printf("Rejecting double-row cluster: dy %f, dz %f, chiY %f, chiZ %f (Y: trk %f prj %f cl %f - Z: trk %f prj %f cl %f)\n", dy, dz, sqrtf(maxDistY), sqrtf(maxDistZ), mP[0], projY, cly, mP[1], projZ, clz));
        if (rejectChi2) {
          clusters[ihit].state |= GPUTPCGMMergedTrackHit::flagRejectDistance;
        }
      } else {
        CADEBUG(printf("\t\tMerging hit row %d X %f Y %f Z %f (dy %f, dz %f, chiY %f, chiZ %f)\n", clusters[ihit].row, clx, cly, clz, dy, dz, sqrtf(maxDistY), sqrtf(maxDistZ)));
        xx += clx * clamp;
        yy += cly * clamp;
        zz += clz * clamp;
        clusterState |= clusters[ihit].state;
        count += clamp;
      }
      if (!(ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].row == clusters[ihit + wayDirection].row && clusters[ihit].slice == clusters[ihit + wayDirection].slice && clusters[ihit].leg == clusters[ihit + wayDirection].leg)) {
        break;
      }
      ihit += wayDirection;
    }
    if (count < 0.1f) {
      CADEBUG(printf("\t\tNo matching cluster in double-row, skipping\n"));
      return -1;
    }
    xx /= count;
    yy /= count;
    zz /= count;
    CADEBUG(printf("\t\tDouble row (%f tot charge)\n", count));
  }
  return 0;
}

GPUd() void GPUTPCGMTrackParam::AttachClusters(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, bool goodLeg, GPUTPCGMPropagator& prop)
{
  float X, Y, Z;
  Merger->GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoX(slice, iRow, mP[0], mP[1], X);
  prop.GetPropagatedYZ(X, Y, Z);
  AttachClusters(Merger, slice, iRow, iTrack, goodLeg, Y, Z);
}

GPUd() void GPUTPCGMTrackParam::AttachClusters(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, bool goodLeg, float Y, float Z)
{
  if (Merger->Param().rec.DisableRefitAttachment & 1) {
    return;
  }
  const GPUTPCTracker& GPUrestrict() tracker = *(Merger->GetConstantMem()->tpcTrackers + slice);
  const GPUTPCRow& GPUrestrict() row = tracker.Row(iRow);
#ifndef GPUCA_TEXTURE_FETCH_CONSTRUCTOR
  GPUglobalref() const cahit2* hits = tracker.HitData(row);
  GPUglobalref() const calink* firsthit = tracker.FirstHitInBin(row);
#endif //! GPUCA_TEXTURE_FETCH_CONSTRUCTOR
  if (row.NHits() == 0) {
    return;
  }

  const float zOffset = Merger->Param().earlyTpcTransform ? ((Merger->OutputTracks()[iTrack].CSide() ^ (slice >= 18)) ? -mTZOffset : mTZOffset) : Merger->GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(slice, mTZOffset, Merger->Param().continuousMaxTimeBin);
  const float y0 = row.Grid().YMin();
  const float stepY = row.HstepY();
  const float z0 = row.Grid().ZMin() - zOffset; // We can use our own ZOffset, since this is only used temporarily anyway
  const float stepZ = row.HstepZ();
  int bin, ny, nz;
  const float tube = 2.5f;
  float nY, nZ;
  Merger->GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoNominalYZ(slice, iRow, Y, Z, nY, nZ);
  row.Grid().GetBinArea(nY, nZ + zOffset, tube, tube, bin, ny, nz);
  float sy2 = tube * tube, sz2 = tube * tube;

  const int nBinsY = row.Grid().Ny();
  const int idOffset = tracker.Data().ClusterIdOffset();
  const int* ids = &(tracker.Data().ClusterDataIndex()[row.HitNumberOffset()]);
  unsigned int myWeight = Merger->TrackOrderAttach()[iTrack] | GPUTPCGMMergerTypes::attachAttached | GPUTPCGMMergerTypes::attachTube;
  GPUAtomic(unsigned int)* const weights = Merger->ClusterAttachment();
  if (goodLeg) {
    myWeight |= GPUTPCGMMergerTypes::attachGoodLeg;
  }
  for (int k = 0; k <= nz; k++) {
    const int mybin = bin + k * nBinsY;
    const unsigned int hitFst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin);
    const unsigned int hitLst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin + ny + 1);
    for (unsigned int ih = hitFst; ih < hitLst; ih++) {
      int id = idOffset + ids[ih];
      GPUAtomic(unsigned int) * GPUrestrict() const weight = weights + id;
      ;
#if !defined(GPUCA_NO_ATOMIC_PRECHECK) && GPUCA_NO_ATOMIC_PRECHECK < 1
      if (myWeight <= *weight) {
        continue;
      }
#endif
      const cahit2 hh = CA_TEXTURE_FETCH(cahit2, gAliTexRefu2, hits, ih);
      const float y = y0 + hh.x * stepY;
      const float z = z0 + hh.y * stepZ;
      const float dy = y - nY;
      const float dz = z - nZ;
      if (dy * dy < sy2 && dz * dz < sz2) {
        // CADEBUG(printf("Found Y %f Z %f\n", y, z));
        CAMath::AtomicMax(weight, myWeight);
      }
    }
  }
}

GPUd() void GPUTPCGMTrackParam::AttachClustersPropagate(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int lastRow, int toRow, int iTrack, bool goodLeg, GPUTPCGMPropagator& GPUrestrict() prop, bool inFlyDirection, float maxSinPhi)
{
  if (Merger->Param().rec.DisableRefitAttachment & 2) {
    return;
  }
  int step = toRow > lastRow ? 1 : -1;
  float xx = mX - Merger->Param().tpcGeometry.Row2X(lastRow);
  for (int iRow = lastRow + step; iRow != toRow; iRow += step) {
    if (CAMath::Abs(mP[2]) > maxSinPhi) {
      return;
    }
    if (CAMath::Abs(mX) > CAMath::Abs(mP[0]) * CAMath::Tan(kSectAngle / 2.f)) {
      return;
    }
    int err = prop.PropagateToXAlpha(xx + Merger->Param().tpcGeometry.Row2X(iRow), prop.GetAlpha(), inFlyDirection);
    if (err) {
      return;
    }
    CADEBUG(printf("Attaching in row %d\n", iRow));
    AttachClusters(Merger, slice, iRow, iTrack, goodLeg, prop);
  }
}

GPUd() bool GPUTPCGMTrackParam::FollowCircleChk(float lrFactor, float toY, float toX, bool up, bool right)
{
  return CAMath::Abs(mX * lrFactor - toY) > 1.f &&                                                                       // transport further in Y
         CAMath::Abs(mP[2]) < 0.7f &&                                                                                    // rotate back
         (up ? (-mP[0] * lrFactor > toX || (right ^ (mP[2] > 0))) : (-mP[0] * lrFactor < toX || (right ^ (mP[2] < 0)))); // don't overshoot in X
}

GPUd() void GPUTPCGMTrackParam::StoreAttachMirror(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, bool goodLeg, float toAlpha, float toY, float toX, int toSlice, int toRow, bool inFlyDirection, float alpha)
{
  unsigned int nLoopData = CAMath::AtomicAdd(&Merger->Memory()->nLoopData, 1u);
  if (nLoopData >= Merger->NMaxTracks()) {
    CAMath::AtomicExch(&Merger->Memory()->nLoopData, 0u);
    // TODO: GPUCA_ERROR_LOOPER_OVERFLOW
    return;
  }
  GPUTPCGMLoopData& data = Merger->LoopData()[nLoopData];
  data.param = *this;
  data.alpha = alpha;
  data.track = iTrack;
  data.goodLeg = goodLeg;
  data.toAlpha = toAlpha;
  data.toY = toY;
  data.toX = toX;
  data.slice = slice;
  data.row = iRow;
  data.toSlice = toSlice;
  data.toRow = toRow;
  data.inFlyDirection = inFlyDirection;
}

GPUd() void GPUTPCGMTrackParam::RefitLoop(const GPUTPCGMMerger* GPUrestrict() Merger, int loopIdx)
{
  GPUTPCGMPropagator prop;
  prop.SetMaterial(kRadLen, kRho);
  prop.SetPolynomialField(&Merger->Param().polynomialField);
  prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
  prop.SetToyMCEventsFlag(Merger->Param().ToyMCEventsFlag);
  prop.SetMatLUT(Merger->GetConstantMem()->calibObjects.matLUT);
  prop.SetSeedingErrors(false);
  prop.SetFitInProjections(true);
  prop.SetPropagateBzOnly(false);

  GPUTPCGMLoopData& data = Merger->LoopData()[loopIdx];
  prop.SetTrack(&data.param, data.alpha);
  if (data.toRow == 0) {
    data.param.AttachClustersMirror(Merger, data.slice, data.row, data.track, data.toY, prop, true);
  } else {
    data.param.FollowCircle(Merger, prop, data.slice, data.row, data.track, data.goodLeg, data.toAlpha, data.toY, data.toX, data.toSlice, data.toRow, data.inFlyDirection, true);
  }
}

GPUd() int GPUTPCGMTrackParam::FollowCircle(const GPUTPCGMMerger* GPUrestrict() Merger, GPUTPCGMPropagator& GPUrestrict() prop, int slice, int iRow, int iTrack, bool goodLeg, float toAlpha, float toX, float toY, int toSlice, int toRow, bool inFlyDirection, bool phase2)
{
  if (Merger->Param().rec.DisableRefitAttachment & 4) {
    return 1;
  }
  if (Merger->Param().rec.loopInterpolationInExtraPass && phase2 == false) {
    StoreAttachMirror(Merger, slice, iRow, iTrack, goodLeg, toAlpha, toY, toX, toSlice, toRow, inFlyDirection, prop.GetAlpha());
    return 1;
  }
  const GPUParam& GPUrestrict() param = Merger->Param();
  bool right;
  float dAlpha = toAlpha - prop.GetAlpha();
  if (CAMath::Abs(dAlpha) > 0.001f) {
    right = CAMath::Abs(dAlpha) < M_PI ? (dAlpha > 0) : (dAlpha < 0);
  } else {
    right = toY > mP[0];
  }
  bool up = (mP[2] < 0) ^ right;
  int targetRow = up ? (GPUCA_ROW_COUNT - 1) : 0;
  float lrFactor = mP[2] > 0 ? 1.f : -1.f; // right ^ down
  // clang-format off
  CADEBUG(printf("CIRCLE Track %d: Slice %d Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f - Next hit: Slice %d Alpha %f X %f Y %f - Right %d Up %d dAlpha %f lrFactor %f\n", iTrack, slice, prop.GetAlpha(), mX, mP[0], mP[1], mP[2], mP[3], toSlice, toAlpha, toX, toY, (int)right, (int)up, dAlpha, lrFactor));
  // clang-format on

  AttachClustersPropagate(Merger, slice, iRow, targetRow, iTrack, goodLeg, prop, inFlyDirection, 0.7f);
  if (prop.RotateToAlpha(prop.GetAlpha() + (M_PI / 2.f) * lrFactor)) {
    return 1;
  }
  CADEBUG(printf("Rotated: X %f Y %f Z %f SinPhi %f (Alpha %f / %f)\n", mP[0], mX, mP[1], mP[2], prop.GetAlpha(), prop.GetAlpha() + M_PI / 2.f));
  while (slice != toSlice || FollowCircleChk(lrFactor, toY, toX, up, right)) {
    while ((slice != toSlice) ? (CAMath::Abs(mX) <= CAMath::Abs(mP[0]) * CAMath::Tan(kSectAngle / 2.f)) : FollowCircleChk(lrFactor, toY, toX, up, right)) {
      int err = prop.PropagateToXAlpha(mX + 1.f, prop.GetAlpha(), inFlyDirection);
      if (err) {
        CADEBUG(printf("propagation error (%d)\n", err));
        prop.RotateToAlpha(prop.GetAlpha() - (M_PI / 2.f) * lrFactor);
        return 1;
      }
      CADEBUG(printf("Propagated to y = %f: X %f Z %f SinPhi %f\n", mX, mP[0], mP[1], mP[2]));
      int found = 0;
      for (int j = 0; j < GPUCA_ROW_COUNT && found < 3; j++) {
        float rowX = Merger->Param().tpcGeometry.Row2X(j);
        if (CAMath::Abs(rowX - (-mP[0] * lrFactor)) < 1.5f) {
          CADEBUG(printf("Attempt row %d (Y %f Z %f)\n", j, mX * lrFactor, mP[1]));
          AttachClusters(Merger, slice, j, iTrack, false, mX * lrFactor, mP[1]);
        }
      }
    }
    if (slice != toSlice) {
      if (right) {
        slice++;
        if (slice >= 18) {
          slice -= 18;
        }
      } else {
        slice--;
        if (slice < 0) {
          slice += 18;
        }
      }
      CADEBUG(printf("Rotating to slice %d\n", slice));
      if (prop.RotateToAlpha(param.Alpha(slice) + (M_PI / 2.f) * lrFactor)) {
        CADEBUG(printf("rotation error\n"));
        prop.RotateToAlpha(prop.GetAlpha() - (M_PI / 2.f) * lrFactor);
        return 1;
      }
      CADEBUG(printf("After Rotatin Alpha %f Position X %f Y %f Z %f SinPhi %f\n", prop.GetAlpha(), mP[0], mX, mP[1], mP[2]));
    }
  }
  CADEBUG(printf("Rotating back\n"));
  for (int i = 0; i < 2; i++) {
    if (prop.RotateToAlpha(prop.GetAlpha() + (M_PI / 2.f) * lrFactor) == 0) {
      break;
    }
    if (i) {
      CADEBUG(printf("Final rotation failed\n"));
      return 1;
    }
    CADEBUG(printf("resetting physical model\n"));
    prop.SetTrack(this, prop.GetAlpha());
  }
  prop.Rotate180();
  CADEBUG(printf("Mirrored position: Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f\n", prop.GetAlpha(), mX, mP[0], mP[1], mP[2], mP[3]));
  iRow = toRow;
  float dx = toX - Merger->Param().tpcGeometry.Row2X(toRow);
  if (up ^ (toX > mX)) {
    if (up) {
      while (iRow < GPUCA_ROW_COUNT - 2 && Merger->Param().tpcGeometry.Row2X(iRow + 1) + dx <= mX) {
        iRow++;
      }
    } else {
      while (iRow > 1 && Merger->Param().tpcGeometry.Row2X(iRow - 1) + dx >= mX) {
        iRow--;
      }
    }
    prop.PropagateToXAlpha(Merger->Param().tpcGeometry.Row2X(iRow) + dx, prop.GetAlpha(), inFlyDirection);
    AttachClustersPropagate(Merger, slice, iRow, toRow, iTrack, !goodLeg, prop, inFlyDirection);
  }
  if (prop.PropagateToXAlpha(toX, prop.GetAlpha(), inFlyDirection)) {
    mX = toX;
  }
  CADEBUG(printf("Final position: Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f\n", prop.GetAlpha(), mX, mP[0], mP[1], mP[2], mP[3]));
  return (0);
}

GPUd() void GPUTPCGMTrackParam::AttachClustersMirror(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, float toY, GPUTPCGMPropagator& GPUrestrict() prop, bool phase2)
{
  if (Merger->Param().rec.DisableRefitAttachment & 8) {
    return;
  }
  if (Merger->Param().rec.loopInterpolationInExtraPass && phase2 == false) {
    StoreAttachMirror(Merger, slice, iRow, iTrack, false, 0, toY, 0, 0, 0, 0, prop.GetAlpha());
    return;
  }
  float X = mP[2] > 0 ? mP[0] : -mP[0];
  float toX = mP[2] > 0 ? toY : -toY;
  float Y = mP[2] > 0 ? -mX : mX;
  float Z = mP[1];
  if (CAMath::Abs(mP[2]) >= GPUCA_MAX_SIN_PHI_LOW) {
    return;
  }
  float SinPhi = CAMath::Sqrt(1 - mP[2] * mP[2]) * (mP[2] > 0 ? -1 : 1);
  if (CAMath::Abs(SinPhi) >= GPUCA_MAX_SIN_PHI_LOW) {
    return;
  }
  float b = prop.GetBz(prop.GetAlpha(), mX, mP[0], mP[1]);

  int count = CAMath::Abs((toX - X) / 0.5f) + 0.5f;
  if (count == 0) {
    return;
  }
  float dx = (toX - X) / count;
  const float myRowX = Merger->Param().tpcGeometry.Row2X(iRow);
  // printf("AttachMirror\n");
  // printf("X %f Y %f Z %f SinPhi %f toY %f -->\n", mX, mP[0], mP[1], mP[2], toY);
  // printf("X %f Y %f Z %f SinPhi %f, count %d dx %f (to: %f)\n", X, Y, Z, SinPhi, count, dx, X + count * dx);
  while (count--) {
    float ex = CAMath::Sqrt(1 - SinPhi * SinPhi);
    float exi = 1.f / ex;
    float dxBzQ = dx * -b * mP[4];
    float newSinPhi = SinPhi + dxBzQ;
    if (CAMath::Abs(newSinPhi) > GPUCA_MAX_SIN_PHI_LOW) {
      return;
    }
    float dS = dx * exi;
    float h2 = dS * exi * exi;
    float h4 = .5f * h2 * dxBzQ;

    X += dx;
    Y += dS * SinPhi + h4;
    Z += dS * mP[3];
    SinPhi = newSinPhi;
    if (CAMath::Abs(X) > CAMath::Abs(Y) * CAMath::Tan(kSectAngle / 2.f)) {
      continue;
    }

    // printf("count %d: At X %f Y %f Z %f SinPhi %f\n", count, mP[2] > 0 ? -Y : Y, mP[2] > 0 ? X : -X, Z, SinPhi);

    float paramX = mP[2] > 0 ? -Y : Y;
    int step = paramX >= mX ? 1 : -1;
    int found = 0;
    for (int j = iRow; j >= 0 && j < GPUCA_ROW_COUNT && found < 3; j += step) {
      float rowX = mX + Merger->Param().tpcGeometry.Row2X(j) - myRowX;
      if (CAMath::Abs(rowX - paramX) < 1.5f) {
        // printf("Attempt row %d\n", j);
        AttachClusters(Merger, slice, j, iTrack, false, mP[2] > 0 ? X : -X, Z);
      }
    }
  }
}

GPUd() void GPUTPCGMTrackParam::ShiftZ2(const GPUTPCGMMergedTrackHit* clusters, const GPUTPCGMMerger* merger, int N)
{
  float tzInner, tzOuter;
  if (merger->Param().earlyTpcTransform) {
    tzInner = clusters[N - 1].z;
    tzOuter = clusters[0].z;
  } else {
    const auto& GPUrestrict() cls = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear;
    tzInner = cls[clusters[N - 1].num].getTime();
    tzOuter = cls[clusters[0].num].getTime();
  }
  ShiftZ(merger, clusters[0].slice, tzInner, tzOuter);
}

GPUd() void GPUTPCGMTrackParam::ShiftZ(const GPUTPCGMMerger* GPUrestrict() merger, int slice, float tz1, float tz2)
{
  if (!merger->Param().ContinuousTracking) {
    return;
  }
  const float cosPhi = CAMath::Abs(mP[2]) < 1.f ? CAMath::Sqrt(1 - mP[2] * mP[2]) : 0.f;
  const float dxf = -CAMath::Abs(mP[2]);
  const float dyf = cosPhi * (mP[2] > 0 ? 1.f : -1.f);
  const float r1 = CAMath::Abs(mP[4] * merger->Param().polynomialField.GetNominalBz());
  const float r = r1 > 0.0001 ? (1.f / CAMath::Abs(r1)) : 10000;
  float xp = mX + dxf * r;
  float yp = mP[0] + dyf * r;
  // printf("X %f Y %f SinPhi %f QPt %f R %f --> XP %f YP %f\n", mX, mP[0], mP[2], mP[4], r, xp, yp);
  const float r2 = (r + CAMath::Sqrt(xp * xp + yp * yp)) / 2.f; // Improve the radius by taking into acount both points we know (00 and xy).
  xp = mX + dxf * r2;
  yp = mP[0] + dyf * r2;
  // printf("X %f Y %f SinPhi %f QPt %f R %f --> XP %f YP %f\n", mX, mP[0], mP[2], mP[4], r2, xp, yp);
  float atana = CAMath::ATan2(CAMath::Abs(xp), CAMath::Abs(yp));
  float atanb = CAMath::ATan2(CAMath::Abs(mX - xp), CAMath::Abs(mP[0] - yp));
  // printf("Tan %f %f (%f %f)\n", atana, atanb, mX - xp, mP[0] - yp);
  const float dS = (xp > 0 ? (atana + atanb) : (atanb - atana)) * r;
  float z0 = dS * mP[3];
  // printf("Track Z %f (Offset %f), z0 %f, V %f (dS %f, dZds %f, qPt %f)             - Z span %f to %f: diff %f\n", mP[1], mTZOffset, z0, mP[1] - z0, dS, mP[3], mP[4], tz2, tz1, tz2 - tz1);
  if (CAMath::Abs(z0) > 250.f) {
    z0 = z0 > 0 ? 250.f : -250.f;
  }
  float deltaZ = mP[1] - z0;
  if (merger->Param().earlyTpcTransform) {
    mTZOffset += deltaZ;
    mP[1] -= deltaZ;
    deltaZ = 0;
    float zMax = CAMath::Max(tz1, tz2);
    float zMin = CAMath::Min(tz1, tz2);
    if (zMin < 0 && zMin - mTZOffset < -250) {
      deltaZ = zMin - mTZOffset + 250;
    } else if (zMax > 0 && zMax - mTZOffset > 250) {
      deltaZ = zMax - mTZOffset - 250;
    }
    if (zMin < 0 && zMax - mTZOffset > 0) {
      deltaZ = zMax - mTZOffset;
    } else if (zMax > 0 && zMin - mTZOffset < 0) {
      deltaZ = zMin - mTZOffset;
    }
    // if (deltaZ != 0) printf("Moving clusters to TPC Range: Side %f, Shift %f: %f to %f --> %f to %f\n", tz2, deltaZ, tz2 - mTZOffset, tz1 - mTZOffset, tz2 - mTZOffset - deltaZ, tz1 - mTZOffset - deltaZ);
    mTZOffset += deltaZ;
    mP[1] -= deltaZ;
  } else {
    float deltaT = merger->GetConstantMem()->calibObjects.fastTransform->convDeltaZtoDeltaTimeInTimeFrame(slice, deltaZ);
    mTZOffset += deltaT;
    mP[1] -= deltaZ;
    const float minT = CAMath::Min(tz1, tz2);
    const float maxT = CAMath::Max(CAMath::Max(tz1, tz2) - merger->GetConstantMem()->calibObjects.fastTransform->getMaxDriftTime(slice), 0.f);
    // printf("T Check: max %f min %f (min2 %f) vtx %f\n", maxT, minT, CAMath::Min(tzinner, tz2), mTZOffset);
    deltaT = 0.f;
    if (mTZOffset < maxT) {
      deltaT = maxT - mTZOffset;
    }
    if (mTZOffset > minT) {
      deltaT = minT - mTZOffset;
    }
    if (deltaT != 0.f) {
      deltaZ = merger->GetConstantMem()->calibObjects.fastTransform->convDeltaTimeToDeltaZinTimeFrame(slice, deltaT);
      mTZOffset += deltaT;
      mP[1] -= deltaZ;
    }
  }
  // printf("\n");
}

GPUd() bool GPUTPCGMTrackParam::CheckCov() const
{
  const float* c = mC;
  bool ok = c[0] >= 0 && c[2] >= 0 && c[5] >= 0 && c[9] >= 0 && c[14] >= 0 && (c[1] * c[1] <= c[2] * c[0]) && (c[3] * c[3] <= c[5] * c[0]) && (c[4] * c[4] <= c[5] * c[2]) && (c[6] * c[6] <= c[9] * c[0]) && (c[7] * c[7] <= c[9] * c[2]) && (c[8] * c[8] <= c[9] * c[5]) &&
            (c[10] * c[10] <= c[14] * c[0]) && (c[11] * c[11] <= c[14] * c[2]) && (c[12] * c[12] <= c[14] * c[5]) && (c[13] * c[13] <= c[14] * c[9]);
  return ok;
}

GPUd() bool GPUTPCGMTrackParam::CheckNumericalQuality(float overrideCovYY) const
{
  //* Check that the track parameters and covariance matrix are reasonable
  bool ok = CAMath::Finite(mX) && CAMath::Finite(mChi2);
  CADEBUG(
    printf("OK %d - ", (int)ok); for (int i = 0; i < 5; i++) { printf("%f ", mP[i]); } printf(" - "); for (int i = 0; i < 15; i++) { printf("%f ", mC[i]); } printf("\n"));
  const float* c = mC;
  for (int i = 0; i < 15; i++) {
    ok = ok && CAMath::Finite(c[i]);
  }
  CADEBUG(printf("OK1 %d\n", (int)ok));
  for (int i = 0; i < 5; i++) {
    ok = ok && CAMath::Finite(mP[i]);
  }
  CADEBUG(printf("OK2 %d\n", (int)ok));
  if ((overrideCovYY > 0 ? overrideCovYY : c[0]) > 4.f * 4.f || c[2] > 4.f * 4.f || c[5] > 2.f * 2.f || c[9] > 2.f * 2.f) {
    ok = 0;
  }
  CADEBUG(printf("OK3 %d\n", (int)ok));
  if (CAMath::Abs(mP[2]) > GPUCA_MAX_SIN_PHI) {
    ok = 0;
  }
  CADEBUG(printf("OK4 %d\n", (int)ok));
  if (!CheckCov()) {
    ok = false;
  }
  CADEBUG(printf("OK5 %d\n", (int)ok));
  return ok;
}

#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)
bool GPUTPCGMTrackParam::GetExtParam(AliExternalTrackParam& T, double alpha) const
{
  //* Convert from GPUTPCGMTrackParam to AliExternalTrackParam parameterisation,
  //* the angle alpha is the global angle of the local X axis

  bool ok = CheckNumericalQuality();

  double par[5], cov[15];
  for (int i = 0; i < 5; i++) {
    par[i] = mP[i];
  }
  for (int i = 0; i < 15; i++) {
    cov[i] = mC[i];
  }

  if (par[2] > GPUCA_MAX_SIN_PHI) {
    par[2] = GPUCA_MAX_SIN_PHI;
  }
  if (par[2] < -GPUCA_MAX_SIN_PHI) {
    par[2] = -GPUCA_MAX_SIN_PHI;
  }

  if (CAMath::Abs(par[4]) < 1.e-5) {
    par[4] = 1.e-5; // some other software will crash if q/Pt==0
  }
  if (CAMath::Abs(par[4]) > 1. / 0.08) {
    ok = 0; // some other software will crash if q/Pt is too big
  }
  T.Set((double)mX, alpha, par, cov);
  return ok;
}

void GPUTPCGMTrackParam::SetExtParam(const AliExternalTrackParam& T)
{
  //* Convert from AliExternalTrackParam parameterisation

  for (int i = 0; i < 5; i++) {
    mP[i] = T.GetParameter()[i];
  }
  for (int i = 0; i < 15; i++) {
    mC[i] = T.GetCovariance()[i];
  }
  mX = T.GetX();
  if (mP[2] > GPUCA_MAX_SIN_PHI) {
    mP[2] = GPUCA_MAX_SIN_PHI;
  }
  if (mP[2] < -GPUCA_MAX_SIN_PHI) {
    mP[2] = -GPUCA_MAX_SIN_PHI;
  }
}
#endif

GPUd() void GPUTPCGMTrackParam::RefitTrack(GPUTPCGMMergedTrack& GPUrestrict() track, int iTrk, const GPUTPCGMMerger* GPUrestrict() merger, GPUTPCGMMergedTrackHit* GPUrestrict() clusters, int attempt) // TODO: Inline me, once __forceinline__ is fixed by HIP
{
  if (!track.OK()) {
    return;
  }

  // clang-format off
  CADEBUG(if (DEBUG_SINGLE_TRACK >= 0 && iTrk != DEBUG_SINGLE_TRACK) { track.SetNClusters(0); track.SetOK(0); return; } );
  // clang-format on

  int nTrackHits = track.NClusters();
  int NTolerated = 0; // Clusters not fit but tollerated for track length cut
  GPUTPCGMTrackParam t = track.Param();
  float Alpha = track.Alpha();
  CADEBUG(int nTrackHitsOld = nTrackHits; float ptOld = t.QPt());
  bool ok = t.Fit(merger, iTrk, clusters + track.FirstClusterRef(), nTrackHits, NTolerated, Alpha, attempt, GPUCA_MAX_SIN_PHI, &track.OuterParam(), merger->Param().dodEdx ? &track.dEdxInfo() : nullptr);
  CADEBUG(printf("Finished Fit Track %d\n", iTrk));
  CADEBUG(printf("OUTPUT hits %d -> %d+%d = %d, QPt %f -> %f, SP %f, ok %d chi2 %f chi2ndf %f\n", nTrackHitsOld, nTrackHits, NTolerated, nTrackHits + NTolerated, ptOld, t.QPt(), t.SinPhi(), (int)ok, t.Chi2(), t.Chi2() / CAMath::Max(1, nTrackHits)));

  if (!ok && attempt == 0 && merger->Param().rec.retryRefit) {
    for (unsigned int i = 0; i < track.NClusters(); i++) {
      clusters[track.FirstClusterRef() + i].state &= GPUTPCGMMergedTrackHit::clustererAndSharedFlags;
    }
    CADEBUG(printf("Track rejected, marking for retry\n"));
    if (merger->Param().rec.retryRefit == 2) {
      nTrackHits = track.NClusters();
      NTolerated = 0; // Clusters not fit but tollerated for track length cut
      t = track.Param();
      Alpha = track.Alpha();
      ok = t.Fit(merger, iTrk, clusters + track.FirstClusterRef(), nTrackHits, NTolerated, Alpha, 1, GPUCA_MAX_SIN_PHI, &track.OuterParam(), merger->Param().dodEdx ? &track.dEdxInfo() : nullptr);
    } else {
      unsigned int nRefit = CAMath::AtomicAdd(&merger->Memory()->nRetryRefit, 1u);
      merger->RetryRefitIds()[nRefit] = iTrk;
      return;
    }
  }
  if (CAMath::Abs(t.QPt()) < 1.e-4f) {
    t.QPt() = 1.e-4f;
  }

  track.SetOK(ok);
  track.SetNClustersFitted(nTrackHits);
  track.Param() = t;
  track.Alpha() = Alpha;

  if (track.OK()) {
    int ind = track.FirstClusterRef();
    const GPUParam& GPUrestrict() param = merger->Param();
    float alphaa = param.Alpha(clusters[ind].slice);
    float xx, yy, zz;
    if (merger->Param().earlyTpcTransform) {
      xx = clusters[ind].x;
      yy = clusters[ind].y;
      zz = clusters[ind].z - track.Param().GetTZOffset();
    } else {
      const ClusterNative& GPUrestrict() cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ind].num];
      merger->GetConstantMem()->calibObjects.fastTransform->Transform(clusters[ind].slice, clusters[ind].row, cl.getPad(), cl.getTime(), xx, yy, zz, track.Param().GetTZOffset());
    }
    float sinA = CAMath::Sin(alphaa - track.Alpha());
    float cosA = CAMath::Cos(alphaa - track.Alpha());
    track.SetLastX(xx * cosA - yy * sinA);
    track.SetLastY(xx * sinA + yy * cosA);
    track.SetLastZ(zz);
  }
}

GPUd() bool GPUTPCGMTrackParam::Rotate(float alpha)
{
  float cA = CAMath::Cos(alpha);
  float sA = CAMath::Sin(alpha);
  float x0 = mX;
  float sinPhi0 = mP[2], cosPhi0 = CAMath::Sqrt(1 - mP[2] * mP[2]);
  float cosPhi = cosPhi0 * cA + sinPhi0 * sA;
  float sinPhi = -cosPhi0 * sA + sinPhi0 * cA;
  float j0 = cosPhi0 / cosPhi;
  float j2 = cosPhi / cosPhi0;
  mX = x0 * cA + mP[0] * sA;
  mP[0] = -x0 * sA + mP[0] * cA;
  mP[2] = sinPhi + j2;
  mC[0] *= j0 * j0;
  mC[1] *= j0;
  mC[3] *= j0;
  mC[6] *= j0;
  mC[10] *= j0;

  mC[3] *= j2;
  mC[4] *= j2;
  mC[5] *= j2 * j2;
  mC[8] *= j2;
  mC[12] *= j2;
  if (cosPhi < 0) { // change direction ( t0 direction is already changed in t0.UpdateValues(); )
    SinPhi() = -SinPhi();
    DzDs() = -DzDs();
    QPt() = -QPt();
    mC[3] = -mC[3];
    mC[4] = -mC[4];
    mC[6] = -mC[6];
    mC[7] = -mC[7];
    mC[10] = -mC[10];
    mC[11] = -mC[11];
  }
  return true;
}
