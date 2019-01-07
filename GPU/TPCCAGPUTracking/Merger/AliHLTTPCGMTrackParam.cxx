// $Id: AliHLTTPCGMTrackParam.cxx 41769 2010-06-16 13:58:00Z sgorbuno $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#define GPUCA_CADEBUG 0
#define DEBUG_SINGLE_TRACK -1

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCGMPropagator.h"
#include "AliHLTTPCGMBorderTrack.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCGMPolynomialField.h"
#include "AliHLTTPCGMMerger.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAClusterData.h"
#ifndef GPUCA_STANDALONE
#include "AliExternalTrackParam.h"
#endif
#include "AliGPUCAParam.h"
#include "AliHLTTPCCAClusterErrorStat.h"
#ifdef GPUCA_CADEBUG_ENABLED
#include "../cmodules/qconfig.h"
#endif

#include <cmath>
#include <stdlib.h>

CADEBUG(int cadebug_nTracks = 0;)

static constexpr float kRho = 1.025e-3;//0.9e-3;
static constexpr float kRadLen = 29.532;//28.94;
static constexpr float kDeg2Rad = M_PI / 180.f;
static constexpr float kSectAngle = 2 * M_PI / 18.f;

GPUd() bool AliHLTTPCGMTrackParam::Fit(const AliHLTTPCGMMerger *merger, int iTrk, AliHLTTPCGMMergedTrackHit *clusters, int &N, int &NTolerated, float &Alpha, int attempt, float maxSinPhi, AliHLTTPCCAOuterParam *outerParam)
{
	const AliGPUCAParam &param = merger->SliceParam();

	AliHLTTPCCAClusterErrorStat errorStat(N);

	AliHLTTPCGMPropagator prop;
	prop.SetMaterial(kRadLen, kRho);
	prop.SetPolynomialField(merger->pField());
	prop.SetMaxSinPhi(maxSinPhi);
	prop.SetToyMCEventsFlag(param.ToyMCEventsFlag);
	ShiftZ(merger->pField(), clusters, param, N);

	int nWays = param.rec.NWays;
	int maxN = N;
	int ihitStart = 0;
	float covYYUpd = 0.;
	float lastUpdateX = -1.;
	unsigned char lastRow = 255;
	unsigned char lastSlice = 255;

	for (int iWay = 0; iWay < nWays; iWay++)
	{
		int nMissed = 0, nMissed2 = 0;
		if (iWay && param.rec.NWaysOuter && iWay == nWays - 1 && outerParam)
		{
			for (int i = 0; i < 5; i++) outerParam->fP[i] = fP[i];
			outerParam->fP[1] += fZOffset;
			for (int i = 0; i < 15; i++) outerParam->fC[i] = fC[i];
			outerParam->fX = fX;
			outerParam->fAlpha = prop.GetAlpha();
		}

	int resetT0 = CAMath::Max(10.f, CAMath::Min(40.f, 150.f / fP[4]));
	const bool refit = ( nWays == 1 || iWay >= 1 );
	const float maxSinForUpdate = CAMath::Sin(70.*kDeg2Rad);
	if (refit && attempt == 0) prop.SetSpecialErrors(true);

	ResetCovariance();
	prop.SetFitInProjections(iWay != 0);
	prop.SetTrack( this, iWay ? prop.GetAlpha() : Alpha);
	ConstrainSinPhi(prop.GetFitInProjections() ? 0.95 : GPUCA_MAX_SIN_PHI_LOW);
	CADEBUG(printf("Fitting track %d way %d (sector %d, alpha %f)\n", cadebug_nTracks, iWay, (int) (prop.GetAlpha() / kSectAngle + 0.5) + (fP[1] < 0 ? 18 : 0), prop.GetAlpha());)

	N = 0;
	lastUpdateX = -1;
	const bool inFlyDirection = iWay & 1;
	unsigned char lastLeg = clusters[ihitStart].fLeg;
	const int wayDirection = (iWay & 1) ? -1 : 1;
	int ihit = ihitStart;
	bool noFollowCircle = false, noFollowCircle2 = false;
	int goodRows = 0;
	for(;ihit >= 0 && ihit<maxN;ihit += wayDirection)
	{
		const bool crossCE = lastSlice != 255 && ((lastSlice < 18) ^ (clusters[ihit].fSlice < 18));
		if (crossCE)
		{
			fZOffset = -fZOffset;
			lastSlice = clusters[ihit].fSlice;
			noFollowCircle2 = true;
		}

		float xx = clusters[ihit].fX;
		float yy = clusters[ihit].fY;
		float zz = clusters[ihit].fZ - fZOffset;
		unsigned char clusterState = clusters[ihit].fState;
		const float clAlpha = param.Alpha(clusters[ihit].fSlice);
		CADEBUG(printf("\tHit %3d/%3d Row %3d: Cluster Alpha %8.3f %d , X %8.3f - Y %8.3f, Z %8.3f (Missed %d)", ihit, maxN, (int) clusters[ihit].fRow, clAlpha, (int) clusters[ihit].fSlice, xx, yy, zz, nMissed);)
		CADEBUG(AliHLTTPCCAStandaloneFramework &hlt = AliHLTTPCCAStandaloneFramework::Instance();if (configStandalone.resetids && (unsigned int) hlt.GetNMCLabels() > clusters[ihit].fNum))
		CADEBUG({printf(" MC:"); for (int i = 0;i < 3;i++) {int mcId = hlt.GetMCLabels()[clusters[ihit].fNum].fClusterID[i].fMCID; if (mcId >= 0) printf(" %d", mcId);}}printf("\n");)
		if ((param.rec.RejectMode > 0 && nMissed >= param.rec.RejectMode) || nMissed2 >= GPUCA_MERGER_MAXN_MISSED_HARD || clusters[ihit].fState & AliHLTTPCGMMergedTrackHit::flagReject)
		{
			CADEBUG(printf("\tSkipping hit, %d hits rejected, flag %X\n", nMissed, (int) clusters[ihit].fState);)
			if (iWay + 2 >= nWays && !(clusters[ihit].fState & AliHLTTPCGMMergedTrackHit::flagReject)) clusters[ihit].fState |= AliHLTTPCGMMergedTrackHit::flagRejectErr;
			continue;
		}

		const bool allowModification = refit && (iWay == 0 || (((nWays - iWay) & 1) ? (ihit >= CAMath::Min(maxN / 2, 30)) : (ihit <= CAMath::Max(maxN / 2, maxN - 30))));
		int ihitMergeFirst = ihit;
		prop.SetStatErrorCurCluster(&clusters[ihit]);

		if (MergeDoubleRowClusters(ihit, wayDirection, clusters, param, prop, xx, yy, zz, maxN, clAlpha, clusterState, allowModification) == -1) {nMissed++;nMissed2++;continue;}

		bool changeDirection = (clusters[ihit].fLeg - lastLeg) & 1;
		CADEBUG(if(changeDirection) printf("\t\tChange direction\n");)
		CADEBUG(printf("\tLeg %3d%14sTrack   Alpha %8.3f %s, X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", (int) clusters[ihit].fLeg, "", prop.GetAlpha(), (fabs(prop.GetAlpha() - clAlpha) < 0.01 ? "   " : " R!"), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), "", sqrtf(fC[0]), sqrtf(fC[2]), sqrtf(fC[5]), sqrtf(fC[14]), fC[10]);)
		if (allowModification && changeDirection && !noFollowCircle && !noFollowCircle2)
		{
			const AliHLTTPCGMTrackParam backup = *this;
			const float backupAlpha = prop.GetAlpha();
			if (lastRow != 255 && FollowCircle(merger, prop, lastSlice, lastRow, iTrk, clusters[ihit].fLeg == clusters[maxN - 1].fLeg, clAlpha, xx, yy, clusters[ihit].fSlice, clusters[ihit].fRow, inFlyDirection))
			{
				CADEBUG(printf("Error during follow circle, resetting track!\n");)
				*this = backup;
				prop.SetTrack(this, backupAlpha);
				noFollowCircle = true;
			}
			else
			{
				MirrorTo(prop, yy, zz, inFlyDirection, param, clusters[ihit].fRow, clusterState, false);
				lastUpdateX = fX;
				lastLeg = clusters[ihit].fLeg;
				lastSlice = clusters[ihit].fSlice;
				lastRow = 255;
				N++;
				resetT0 = CAMath::Max(10.f, CAMath::Min(40.f, 150.f / fP[4]));
				CADEBUG(printf("\n");)
				CADEBUG(printf("\t%21sMirror  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), "", sqrtf(fC[0]), sqrtf(fC[2]), sqrtf(fC[5]), sqrtf(fC[14]), fC[10]);)
				continue;
			}
		}
		else if (allowModification && lastRow != 255 && abs(clusters[ihit].fRow - lastRow) > 1)
		{
			AttachClustersPropagate(merger, clusters[ihit].fSlice, lastRow, clusters[ihit].fRow, iTrk, clusters[ihit].fLeg == clusters[maxN - 1].fLeg, prop, inFlyDirection);
		}

		int err = prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection);
		CADEBUG(if(!CheckCov()) printf("INVALID COV AFTER PROPAGATE!!!\n");)
		if (err == -2) //Rotation failed, try to bring to new x with old alpha first, rotate, and then propagate to x, alpha
		{
			CADEBUG(printf("REROTATE\n");)
			if (prop.PropagateToXAlpha(xx, prop.GetAlpha(), inFlyDirection ) == 0)
				err = prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection );
		}
		if (lastRow == 255 || abs((int) lastRow - (int) clusters[ihit].fRow) > 5 || lastSlice != clusters[ihit].fSlice || (param.rec.RejectMode < 0 && -nMissed <= param.rec.RejectMode)) goodRows = 0;
		else goodRows++;
		if (err == 0)
		{
			lastRow = clusters[ihit].fRow;
			lastSlice = clusters[ihit].fSlice;
		}

		CADEBUG(printf("\t%21sPropaga Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f)   ---   Res %8.3f %8.3f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   Err %d", "", prop.GetAlpha(), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), fP[0] - yy, fP[1] - zz, sqrtf(fC[0]), sqrtf(fC[2]), sqrtf(fC[5]), sqrtf(fC[14]), fC[10], err);)

		if (err == 0 && changeDirection)
		{
			const float mirrordY = prop.GetMirroredYTrack();
			CADEBUG(printf(" -- MiroredY: %f --> %f", fP[0], mirrordY);)
			if (fabs(yy - fP[0]) > fabs(yy - mirrordY))
			{
				CADEBUG(printf(" - Mirroring!!!");)
				if (allowModification) AttachClustersMirror(merger, clusters[ihit].fSlice, clusters[ihit].fRow, iTrk, yy, prop); //Never true, will always call FollowCircle above
				MirrorTo(prop, yy, zz, inFlyDirection, param, clusters[ihit].fRow, clusterState, true);
				noFollowCircle = false;

				lastUpdateX = fX;
				lastLeg = clusters[ihit].fLeg;
				lastRow = 255;
				N++;
				resetT0 = CAMath::Max(10.f, CAMath::Min(40.f, 150.f / fP[4]));
				CADEBUG(printf("\n");)
				CADEBUG(printf("\t%21sMirror  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), "", sqrtf(fC[0]), sqrtf(fC[2]), sqrtf(fC[5]), sqrtf(fC[14]), fC[10]);)
				continue;
			}
		}

		if (allowModification) AttachClusters(merger, clusters[ihit].fSlice, clusters[ihit].fRow, iTrk, clusters[ihit].fLeg == clusters[maxN - 1].fLeg);

		const int err2 = fNDF > 0 && CAMath::Abs(prop.GetSinPhi0()) >= maxSinForUpdate;
		if ( err || err2 )
		{
			if (fP[0] > GPUCA_MERGER_COV_LIMIT || fP[2] > GPUCA_MERGER_COV_LIMIT) break;
			MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, AliHLTTPCGMMergedTrackHit::flagNotFit);
			nMissed2++;
			NTolerated++;
			CADEBUG(printf(" --- break (%d, %d)\n", err, err2);)
			continue;
		}
		CADEBUG(printf("\n");)
		errorStat.Fill(xx, yy, zz, prop.GetAlpha(), fX, fP, fC, ihit, iWay);

		int retVal;
		float threshold = 3. + (lastUpdateX >= 0 ? (fabs(fX - lastUpdateX) / 2) : 0.);
		if (fNDF > 5 && (fabs(yy - fP[0]) > threshold || fabs(zz - fP[1]) > threshold)) retVal = 2;
		else retVal = prop.Update( yy, zz, clusters[ihit].fRow, param, clusterState, allowModification && goodRows > 5, refit);
		CADEBUG(if(!CheckCov()) printf("INVALID COV AFTER UPDATE!!!\n");)
		CADEBUG(printf("\t%21sFit     Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f), DzDs %5.2f %16s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   Err %d\n", "", prop.GetAlpha(), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), fP[3], "", sqrtf(fC[0]), sqrtf(fC[2]), sqrtf(fC[5]), sqrtf(fC[14]), fC[10], retVal);)

		if (retVal == 0) // track is updated
		{
			noFollowCircle2 = false;
			lastUpdateX = fX;
			covYYUpd = fC[0];
			nMissed = nMissed2 = 0;
			UnmarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, AliHLTTPCGMMergedTrackHit::flagNotFit);
			N++;
			ihitStart = ihit;
			float dy = fP[0] - prop.Model().Y();
			float dz = fP[1] - prop.Model().Z();
			if (CAMath::Abs(fP[4]) > 10 && --resetT0 <= 0 && CAMath::Abs(fP[2]) < 0.15 && dy*dy+dz*dz>1)
			{
				CADEBUG(printf("Reinit linearization\n");)
				prop.SetTrack(this, prop.GetAlpha());
			}
		}
		else if (retVal == 2) // cluster far away form the track
		{
			if (allowModification) MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, AliHLTTPCGMMergedTrackHit::flagRejectDistance);
			nMissed++;
			nMissed2++;
		}
		else break; // bad chi2 for the whole track, stop the fit
		}
		if (((nWays - iWay) & 1)) ShiftZ(merger->pField(), clusters, param, N);
	}
	ConstrainSinPhi();

	bool ok = N + NTolerated >= TRACKLET_SELECTOR_MIN_HITS(fP[4]) && CheckNumericalQuality(covYYUpd);
	if (!ok) return(false);

	Alpha = prop.GetAlpha();
	if (param.rec.TrackReferenceX <= 500)
	{
		for (int k = 0;k < 3;k++) //max 3 attempts
		{
			int err = prop.PropagateToXAlpha(param.rec.TrackReferenceX, Alpha, 0);
			ConstrainSinPhi();
			if (fabs(fP[0]) <= fX * tanf(kSectAngle / 2.f)) break;
			float dAngle = floor(atan2f(fP[0], fX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
			Alpha += dAngle;
			if (err || k == 2)
			{
				Rotate(dAngle);
				ConstrainSinPhi();
				break;
		}
	}

	}
	else if (fabs(fP[0]) > fX * tanf(kSectAngle / 2.f))
	{
		float dAngle = floor(atan2f(fP[0], fX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
		Rotate(dAngle);
		ConstrainSinPhi();
		Alpha += dAngle;
	}
	if (Alpha > M_PI) Alpha -= 2 * M_PI;
	else if (Alpha <= -M_PI) Alpha += 2 * M_PI;

	return(ok);
}

GPUd() void AliHLTTPCGMTrackParam::MirrorTo(AliHLTTPCGMPropagator& prop, float toY, float toZ, bool inFlyDirection, const AliGPUCAParam& param, unsigned char row, unsigned char clusterState, bool mirrorParameters)
{
	if (mirrorParameters) prop.Mirror(inFlyDirection);
	float err2Y, err2Z;
	prop.GetErr2(err2Y, err2Z, param, toZ, row, clusterState);
	prop.Model().Y() = fP[0] = toY;
	prop.Model().Z() = fP[1] = toZ;
	if (fC[0] < err2Y) fC[0] = err2Y;
	if (fC[2] < err2Z) fC[2] = err2Z;
	if (fabs(fC[5]) < 0.1) fC[5] = fC[5] > 0 ? 0.1 : -0.1;
	if (fC[9] < 1.) fC[9] = 1.;
	fC[1] = fC[4] = fC[6] = fC[8] = fC[11] = fC[13] = 0;
	prop.SetTrack(this, prop.GetAlpha());
	fNDF = -3;
	fChi2 = 0;
}

GPUd() int AliHLTTPCGMTrackParam::MergeDoubleRowClusters(int ihit, int wayDirection, AliHLTTPCGMMergedTrackHit* clusters, const AliGPUCAParam &param, AliHLTTPCGMPropagator& prop, float& xx, float& yy, float& zz, int maxN, float clAlpha, unsigned char& clusterState, bool rejectChi2)
{
	if (ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].fRow == clusters[ihit + wayDirection].fRow && clusters[ihit].fSlice == clusters[ihit + wayDirection].fSlice && clusters[ihit].fLeg == clusters[ihit + wayDirection].fLeg)
	{
		float maxDistY, maxDistZ;
		prop.GetErr2(maxDistY, maxDistZ, param, zz, clusters[ihit].fRow, 0);
		maxDistY = (maxDistY + fC[0]) * 20.f;
		maxDistZ = (maxDistZ + fC[2]) * 20.f;
		int noReject = 0; //Cannot reject if simple estimation of y/z fails (extremely unlike case)
		if (fabs(clAlpha - prop.GetAlpha()) > 1.e-4) noReject = prop.RotateToAlpha(clAlpha);
		float projY, projZ;
		if (noReject == 0) noReject |= prop.GetPropagatedYZ(xx, projY, projZ);
		float count = 0.f;
		xx = yy = zz = 0.f;
		clusterState = 0;
		while(true)
		{
			float dy = clusters[ihit].fY - projY, dz = clusters[ihit].fZ - projZ;
			if (noReject == 0 && (dy * dy > maxDistY || dz * dz > maxDistZ))
			{
				CADEBUG(printf("Rejecting double-row cluster: dy %f, dz %f, chiY %f, chiZ %f (Y: trk %f prj %f cl %f - Z: trk %f prj %f cl %f)\n", dy, dz, sqrtf(maxDistY), sqrtf(maxDistZ), fP[0], projY, clusters[ihit].fY, fP[1], projZ, clusters[ihit].fZ);)
				if (rejectChi2) clusters[ihit].fState |= AliHLTTPCGMMergedTrackHit::flagRejectDistance;
			}
			else
			{
				CADEBUG(printf("\t\tMerging hit row %d X %f Y %f Z %f (dy %f, dz %f, chiY %f, chiZ %f)\n", clusters[ihit].fRow, clusters[ihit].fX, clusters[ihit].fY, clusters[ihit].fZ, dy, dz, sqrtf(maxDistY), sqrtf(maxDistZ));)
				const float amp = clusters[ihit].fAmp;
				xx += clusters[ihit].fX * amp;
				yy += clusters[ihit].fY * amp;
				zz += (clusters[ihit].fZ - fZOffset) * amp;
				clusterState |= clusters[ihit].fState;
				count += amp;
			}
			if (!(ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].fRow == clusters[ihit + wayDirection].fRow && clusters[ihit].fSlice == clusters[ihit + wayDirection].fSlice && clusters[ihit].fLeg == clusters[ihit + wayDirection].fLeg)) break;
			ihit += wayDirection;
		}
		if (count < 0.1)
		{
			CADEBUG(printf("\t\tNo matching cluster in double-row, skipping\n");)
			return -1;
		}
		xx /= count;
		yy /= count;
		zz /= count;
		CADEBUG(printf("\t\tDouble row (%f tot charge)\n", count);)
	}
	return 0;
}

GPUd() void AliHLTTPCGMTrackParam::AttachClusters(const AliHLTTPCGMMerger *Merger, int slice, int iRow, int iTrack, bool goodLeg)
{
	AttachClusters(Merger, slice, iRow, iTrack, goodLeg, fP[0], fP[1]);
}

GPUd() void AliHLTTPCGMTrackParam::AttachClusters(const AliHLTTPCGMMerger *Merger, int slice, int iRow, int iTrack, bool goodLeg, float Y, float Z)
{
#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_BUILD_O2_LIB)
	const AliHLTTPCCATracker &tracker = *(Merger->SliceTrackers() + slice);
	MAKESharedRef(AliHLTTPCCARow, row, tracker.Row(iRow), s.fRows[iRow]);
#ifndef GPUCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
	GPUglobalref() const cahit2 *hits = tracker.HitData(row);
	GPUglobalref() const calink *firsthit = tracker.FirstHitInBin(row);
#endif //!GPUCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
	if (row.NHits() == 0) return;

	const float y0 = row.Grid().YMin();
	const float stepY = row.HstepY();
	const float z0 = row.Grid().ZMin() - fZOffset; //We can use our own ZOffset, since this is only used temporarily anyway
	const float stepZ = row.HstepZ();
	int bin, ny, nz;
	const float tube = 2.5f;
	row.Grid().GetBinArea(Y, Z + fZOffset, tube, tube, bin, ny, nz);
	float sy2 = tube * tube, sz2 = tube * tube;

	for (int k = 0; k <= nz; k++)
	{
		int nBinsY = row.Grid().Ny();
		int mybin = bin + k * nBinsY;
		unsigned int hitFst = TEXTUREFetchCons(calink, gAliTexRefu, firsthit, mybin);
		unsigned int hitLst = TEXTUREFetchCons(calink, gAliTexRefu, firsthit, mybin + ny + 1);
		for (unsigned int ih = hitFst; ih < hitLst; ih++)
		{
			assert((signed) ih < row.NHits());
			cahit2 hh = TEXTUREFetchCons(cahit2, gAliTexRefu2, hits, ih);
			int id = tracker.ClusterData()->Id(tracker.Data().ClusterDataIndex(row, ih));
			int *weight = &Merger->ClusterAttachment()[id];
			if (*weight & AliHLTTPCGMMerger::attachGood) continue;
			float y = y0 + hh.x * stepY;
			float z = z0 + hh.y * stepZ;
			float dy = y - Y;
			float dz = z - Z;
			if (dy * dy < sy2 && dz * dz < sz2)
			{
				//CADEBUG(printf("Found Y %f Z %f\n", y, z);)
				int myWeight = Merger->TrackOrder()[iTrack] | AliHLTTPCGMMerger::attachAttached | AliHLTTPCGMMerger::attachTube;
				if (goodLeg) myWeight |= AliHLTTPCGMMerger::attachGoodLeg;
				CAMath::AtomicMax(weight, myWeight);
			}
		}
	}
#endif
}

GPUd() void AliHLTTPCGMTrackParam::AttachClustersPropagate(const AliHLTTPCGMMerger *Merger, int slice, int lastRow, int toRow, int iTrack, bool goodLeg, AliHLTTPCGMPropagator &prop, bool inFlyDirection, float maxSinPhi)
{
#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_BUILD_O2_LIB)
	int step = toRow > lastRow ? 1 : -1;
	float xx = fX - Merger->SliceParam().RowX[lastRow];
	for (int iRow = lastRow + step; iRow != toRow; iRow += step)
	{
		if (fabs(fP[2]) > maxSinPhi) return;
		if (fabs(fX) > fabs(fP[0]) * tanf(kSectAngle / 2.f)) return;
		int err = prop.PropagateToXAlpha(xx + Merger->SliceParam().RowX[iRow], prop.GetAlpha(), inFlyDirection);
		if (err) return;
		CADEBUG(printf("Attaching in row %d\n", iRow);)
		AttachClusters(Merger, slice, iRow, iTrack, goodLeg);
	}
#endif
}

GPUd() bool AliHLTTPCGMTrackParam::FollowCircleChk(float lrFactor, float toY, float toX, bool up, bool right)
{
	return fabs(fX * lrFactor - toY) > 1.f &&                                                                              //transport further in Y
	       fabs(fP[2]) < 0.7 &&                                                                                            //rotate back
	       (up ? (-fP[0] * lrFactor > toX || (right ^ (fP[2] > 0))) : (-fP[0] * lrFactor < toX || (right ^ (fP[2] < 0)))); //don't overshoot in X
}

GPUd() int AliHLTTPCGMTrackParam::FollowCircle(const AliHLTTPCGMMerger *Merger, AliHLTTPCGMPropagator &prop, int slice, int iRow, int iTrack, bool goodLeg, float toAlpha, float toX, float toY, int toSlice, int toRow, bool inFlyDirection)
{
#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_BUILD_O2_LIB)
	const AliGPUCAParam &param = Merger->SliceParam();
	bool right;
	float dAlpha = toAlpha - prop.GetAlpha();
	if (fabs(dAlpha) > 0.001)
	{
		right = fabs(dAlpha) < M_PI ? (dAlpha > 0) : (dAlpha < 0);
	}
	else
	{
		right = toY > fP[0];
	}
	bool up = (fP[2] < 0) ^ right;
	int targetRow = up ? (GPUCA_ROW_COUNT - 1) : 0;
	float lrFactor = fP[2] > 0 ? 1.f : -1.f; //right ^ down
	CADEBUG(printf("CIRCLE Track %d: Slice %d Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f - Next hit: Slice %d Alpha %f X %f Y %f - Right %d Up %d dAlpha %f lrFactor %f\n", iTrack, slice, prop.GetAlpha(), fX, fP[0], fP[1], fP[2], fP[3], toSlice, toAlpha, toX, toY, (int) right, (int) up, dAlpha, lrFactor);)

	AttachClustersPropagate(Merger, slice, iRow, targetRow, iTrack, goodLeg, prop, inFlyDirection, 0.7);
	if (prop.RotateToAlpha(prop.GetAlpha() + (M_PI / 2.f) * lrFactor)) return 1;
	CADEBUG(printf("Rotated: X %f Y %f Z %f SinPhi %f (Alpha %f / %f)\n", fP[0], fX, fP[1], fP[2], prop.GetAlpha(), prop.GetAlpha() + M_PI / 2.f);)
	while (slice != toSlice || FollowCircleChk(lrFactor, toY, toX, up, right))
	{
		while ((slice != toSlice) ? (fabs(fX) <= fabs(fP[0]) * tanf(kSectAngle / 2.f)) : FollowCircleChk(lrFactor, toY, toX, up, right))
		{
			int err = prop.PropagateToXAlpha(fX + 1.f, prop.GetAlpha(), inFlyDirection);
			if (err)
			{
				CADEBUG(printf("propagation error (%d)\n", err);)
				prop.RotateToAlpha(prop.GetAlpha() - (M_PI / 2.f) * lrFactor);
				return 1;
			}
			CADEBUG(printf("Propagated to y = %f: X %f Z %f SinPhi %f\n", fX, fP[0], fP[1], fP[2]);)
			int found = 0;
			for (int j = 0; j < GPUCA_ROW_COUNT && found < 3; j++)
			{
				float rowX = Merger->SliceParam().RowX[j];
				if (fabs(rowX - (-fP[0] * lrFactor)) < 1.5)
				{
					CADEBUG(printf("Attempt row %d (Y %f Z %f)\n", j, fX * lrFactor, fP[1]);)
					AttachClusters(Merger, slice, j, iTrack, false, fX * lrFactor, fP[1]);
				}
			}
		}
		if (slice != toSlice)
		{
			if (right) {slice++; if (slice >= 18) slice -= 18;}
			else {slice--; if (slice < 0) slice += 18;}
			CADEBUG(printf("Rotating to slice %d\n", slice);)
			if (prop.RotateToAlpha(param.Alpha(slice) + (M_PI / 2.f) * lrFactor))
			{
				CADEBUG(printf("rotation error\n");)
				prop.RotateToAlpha(prop.GetAlpha() - (M_PI / 2.f) * lrFactor);
				return 1;
			}
			CADEBUG(printf("After Rotatin Alpha %f Position X %f Y %f Z %f SinPhi %f\n", prop.GetAlpha(), fP[0], fX, fP[1], fP[2]);)
		}
	}
	CADEBUG(printf("Rotating back\n");)
	for (int i = 0;i < 2;i++)
	{
		if (prop.RotateToAlpha(prop.GetAlpha() + (M_PI / 2.f) * lrFactor) == 0) break;
		if (i)
		{
			CADEBUG(printf("Final rotation failed\n");)
			return 1 ;
		}
		CADEBUG(printf("resetting physical model\n");)
		prop.SetTrack(this, prop.GetAlpha());
	}
	prop.Rotate180();
	CADEBUG(printf("Mirrored position: Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f\n", prop.GetAlpha(), fX, fP[0], fP[1], fP[2], fP[3]);)
	iRow = toRow;
	float dx = toX - Merger->SliceParam().RowX[toRow];
	if (up ^ (toX > fX))
	{
		if (up) while (iRow < GPUCA_ROW_COUNT - 2 && Merger->SliceParam().RowX[iRow + 1] + dx <= fX) iRow++;
		else while (iRow > 1 && Merger->SliceParam().RowX[iRow - 1] + dx >= fX) iRow--;
		prop.PropagateToXAlpha(Merger->SliceParam().RowX[iRow] + dx, prop.GetAlpha(), inFlyDirection);
		AttachClustersPropagate(Merger, slice, iRow, toRow, iTrack, !goodLeg, prop, inFlyDirection);
	}
	if (prop.PropagateToXAlpha(toX, prop.GetAlpha(), inFlyDirection)) fX = toX;
	CADEBUG(printf("Final position: Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f\n", prop.GetAlpha(), fX, fP[0], fP[1], fP[2], fP[3]);)
	return(0);
#else
	return(1);
#endif
}

GPUd() void AliHLTTPCGMTrackParam::AttachClustersMirror(const AliHLTTPCGMMerger* Merger, int slice, int iRow, int iTrack, float toY, AliHLTTPCGMPropagator& prop)
{
#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_BUILD_O2_LIB)
	float X = fP[2] > 0 ? fP[0] : -fP[0];
	float toX = fP[2] > 0 ? toY : -toY;
	float Y = fP[2] > 0 ? -fX : fX;
	float Z = fP[1];
	if (fabs(fP[2]) >= GPUCA_MAX_SIN_PHI_LOW) return;
	float SinPhi = sqrtf(1 - fP[2] * fP[2]) * (fP[2] > 0 ? -1 : 1);
	if (fabs(SinPhi) >= GPUCA_MAX_SIN_PHI_LOW) return;
	float b = prop.GetBz(prop.GetAlpha(), fX, fP[0], fP[1]);

	int count = fabs((toX - X) / 0.5) + 0.5;
	float dx = (toX - X) / count;
	const float myRowX = Merger->SliceParam().RowX[iRow];
	//printf("AttachMirror\n");
	//printf("X %f Y %f Z %f SinPhi %f toY %f -->\n", fX, fP[0], fP[1], fP[2], toY);
	//printf("X %f Y %f Z %f SinPhi %f, count %d dx %f (to: %f)\n", X, Y, Z, SinPhi, count, dx, X + count * dx);
	while (count--)
	{
		float ex = sqrtf(1 - SinPhi * SinPhi);
		float exi = 1. / ex;
		float dxBzQ = dx * -b * fP[4];
		float newSinPhi = SinPhi + dxBzQ;
		if (fabs(newSinPhi) > GPUCA_MAX_SIN_PHI_LOW) return;
		float dS = dx * exi;
		float h2 = dS * exi * exi;
		float h4 = .5 * h2 * dxBzQ;

		X += dx;
		Y += dS * SinPhi + h4;
		Z += dS * fP[3];
		SinPhi = newSinPhi;
		if (fabs(X) > fabs(Y) * tanf(kSectAngle / 2.f)) continue;

		//printf("count %d: At X %f Y %f Z %f SinPhi %f\n", count, fP[2] > 0 ? -Y : Y, fP[2] > 0 ? X : -X, Z, SinPhi);

		float paramX = fP[2] > 0 ? -Y : Y;
		int step = paramX >= fX ? 1 : -1;
		int found = 0;
		for (int j = iRow;j >= 0 && j < GPUCA_ROW_COUNT && found < 3;j += step)
		{
			float rowX = fX + Merger->SliceParam().RowX[j] - myRowX;
			if (fabs(rowX - paramX) < 1.5)
			{
				//printf("Attempt row %d\n", j);
				AttachClusters(Merger, slice, j, iTrack, false, fP[2] > 0 ? X : -X, Z);
			}
		}
	}
#endif
}

GPUd() void AliHLTTPCGMTrackParam::ShiftZ(const AliHLTTPCGMPolynomialField* field, const AliHLTTPCGMMergedTrackHit* clusters, const AliGPUCAParam &param, int N)
{
	if (!param.ContinuousTracking) return;
	if ((clusters[0].fSlice < 18) ^ (clusters[N - 1].fSlice < 18)) return; //Do not shift tracks crossing the central electrode

	const float cosPhi = fabs(fP[2]) < 1.f ? CAMath::Sqrt(1 - fP[2] * fP[2]) : 0.f;
	const float dxf = -CAMath::Abs(fP[2]);
	const float dyf = cosPhi * (fP[2] > 0 ? 1. : -1.);
	const float r = 1./fabs(fP[4] * field->GetNominalBz());
	float xp = fX + dxf * r;
	float yp = fP[0] + dyf * r;
	//printf("X %f Y %f SinPhi %f QPt %f R %f --> XP %f YP %f\n", fX, fP[0], fP[2], fP[4], r, xp, yp);
	const float r2 = (r + CAMath::Sqrt(xp * xp + yp * yp)) / 2.; //Improve the radius by taking into acount both points we know (00 and xy).
	xp = fX + dxf * r2;
	yp = fP[0] + dyf * r2;
	//printf("X %f Y %f SinPhi %f QPt %f R %f --> XP %f YP %f\n", fX, fP[0], fP[2], fP[4], r2, xp, yp);
	float atana = CAMath::ATan2(CAMath::Abs(xp), CAMath::Abs(yp));
	float atanb = CAMath::ATan2(CAMath::Abs(fX - xp), CAMath::Abs(fP[0] - yp));
	//printf("Tan %f %f (%f %f)\n", atana, atanb, fX - xp, fP[0] - yp);
	const float dS = (xp > 0 ? (atana + atanb) : (atanb - atana)) * r;
	float dz = dS * fP[3];
	//printf("Track Z %f (Offset %f), dz %f, V %f (dS %f, dZds %f, qPt %f)             - Z span %f to %f: diff %f\n", fP[1], fZOffset, dz, fP[1] - dz, dS, fP[3], fP[4], clusters[0].fZ, clusters[N - 1].fZ, clusters[0].fZ - clusters[N - 1].fZ);
	if (CAMath::Abs(dz) > 250.) dz = dz > 0 ? 250. : -250.;
	float dZOffset = fP[1] - dz;
	fZOffset += dZOffset;
	fP[1] -= dZOffset;
	dZOffset = 0;
	float zMax = CAMath::Max(clusters[0].fZ, clusters[N - 1].fZ);
	float zMin = CAMath::Min(clusters[0].fZ, clusters[N - 1].fZ);
	if (zMin < 0 && zMin - fZOffset < -250) dZOffset = zMin - fZOffset + 250;
	else if (zMax > 0 && zMax - fZOffset > 250) dZOffset = zMax - fZOffset - 250;
	if (zMin < 0 && zMax - fZOffset > 0) dZOffset = zMax - fZOffset;
	else if (zMax > 0 && zMin - fZOffset < 0) dZOffset = zMin - fZOffset;
	//if (dZOffset != 0) printf("Moving clusters to TPC Range: Side %f, Shift %f: %f to %f --> %f to %f\n", clusters[0].fZ, dZOffset, clusters[0].fZ - fZOffset, clusters[N - 1].fZ - fZOffset, clusters[0].fZ - fZOffset - dZOffset, clusters[N - 1].fZ - fZOffset - dZOffset);
	fZOffset += dZOffset;
	fP[1] -= dZOffset;
	//printf("\n");
}

GPUd() bool AliHLTTPCGMTrackParam::CheckCov() const
{
	const float *c = fC;
	bool ok = c[0] >= 0 && c[2] >= 0 && c[5] >= 0 && c[9] >= 0 && c[14] >= 0
	&& ( c[1]*c[1]<=c[2]*c[0] )
	&& ( c[3]*c[3]<=c[5]*c[0] )
	&& ( c[4]*c[4]<=c[5]*c[2] )
	&& ( c[6]*c[6]<=c[9]*c[0] )
	&& ( c[7]*c[7]<=c[9]*c[2] )
	&& ( c[8]*c[8]<=c[9]*c[5] )
	&& ( c[10]*c[10]<=c[14]*c[0] )
	&& ( c[11]*c[11]<=c[14]*c[2] )
	&& ( c[12]*c[12]<=c[14]*c[5] )
	&& ( c[13]*c[13]<=c[14]*c[9] );
	return ok;
}

GPUd() bool AliHLTTPCGMTrackParam::CheckNumericalQuality(float overrideCovYY) const
{
	//* Check that the track parameters and covariance matrix are reasonable
	bool ok = CAMath::Finite(fX) && CAMath::Finite( fChi2 );
	CADEBUG(printf("OK %d - ", (int) ok); for (int i = 0;i < 5;i++) printf("%f ", fP[i]); printf(" - "); for (int i = 0;i < 15;i++) printf("%f ", fC[i]); printf("\n");)
	const float *c = fC;
	for ( int i = 0; i < 15; i++ ) ok = ok && CAMath::Finite( c[i] );
	CADEBUG(printf("OK1 %d\n", (int) ok);)
	for ( int i = 0; i < 5; i++ ) ok = ok && CAMath::Finite( fP[i] );
	CADEBUG(printf("OK2 %d\n", (int) ok);)
	if ( (overrideCovYY > 0 ? overrideCovYY : c[0]) > 4.*4. || c[2] > 4.*4. || c[5] > 2.*2. || c[9] > 2.*2. ) ok = 0;
	CADEBUG(printf("OK3 %d\n", (int) ok);)
	if ( fabs( fP[2] ) > GPUCA_MAX_SIN_PHI ) ok = 0;
	CADEBUG(printf("OK4 %d\n", (int) ok);)
	if (!CheckCov()) ok = false;
	CADEBUG(printf("OK5 %d\n", (int) ok);)
	return ok;
}

#if !defined(GPUCA_STANDALONE) & !defined(GPUCA_GPUCODE)
bool AliHLTTPCGMTrackParam::GetExtParam( AliExternalTrackParam &T, double alpha ) const
{
	//* Convert from AliHLTTPCGMTrackParam to AliExternalTrackParam parameterisation,
	//* the angle alpha is the global angle of the local X axis

	bool ok = CheckNumericalQuality();

	double par[5], cov[15];
	for ( int i = 0; i < 5; i++ ) par[i] = fP[i];
	for ( int i = 0; i < 15; i++ ) cov[i] = fC[i];

	if ( par[2] > GPUCA_MAX_SIN_PHI ) par[2] = GPUCA_MAX_SIN_PHI;
	if ( par[2] < -GPUCA_MAX_SIN_PHI ) par[2] = -GPUCA_MAX_SIN_PHI;

	if ( fabs( par[4] ) < 1.e-5 ) par[4] = 1.e-5; // some other software will crash if q/Pt==0
	if ( fabs( par[4] ) > 1./0.08 ) ok = 0; // some other software will crash if q/Pt is too big

	T.Set( (double) fX, alpha, par, cov );
	return ok;
}

void AliHLTTPCGMTrackParam::SetExtParam( const AliExternalTrackParam &T )
{
	//* Convert from AliExternalTrackParam parameterisation

	for ( int i = 0; i < 5; i++ ) fP[i] = T.GetParameter()[i];
	for ( int i = 0; i < 15; i++ ) fC[i] = T.GetCovariance()[i];
	fX = T.GetX();
	if ( fP[2] > GPUCA_MAX_SIN_PHI ) fP[2] = GPUCA_MAX_SIN_PHI;
	if ( fP[2] < -GPUCA_MAX_SIN_PHI ) fP[2] = -GPUCA_MAX_SIN_PHI;
}
#endif

GPUd() void AliHLTTPCGMTrackParam::RefitTrack(AliHLTTPCGMMergedTrack &track, int iTrk, const AliHLTTPCGMMerger* merger, AliHLTTPCGMMergedTrackHit* clusters)
{
	if( !track.OK() ) return;

	CADEBUG(cadebug_nTracks++;)
	CADEBUG(if (DEBUG_SINGLE_TRACK >= 0 && cadebug_nTracks != DEBUG_SINGLE_TRACK) {track.SetNClusters(0);track.SetOK(0);return;})

	const int nAttempts = 2;
	for (int attempt = 0;;)
	{
		int nTrackHits = track.NClusters();
		int NTolerated = 0; //Clusters not fit but tollerated for track length cut
		AliHLTTPCGMTrackParam t = track.Param();
		float Alpha = track.Alpha();
		CADEBUG(int nTrackHitsOld = nTrackHits; float ptOld = t.QPt();)
		bool ok = t.Fit( merger, iTrk, clusters + track.FirstClusterRef(), nTrackHits, NTolerated, Alpha, attempt, GPUCA_MAX_SIN_PHI, &track.OuterParam() );
		CADEBUG(printf("Finished Fit Track %d\n", cadebug_nTracks);)

		if ( fabs( t.QPt() ) < 1.e-4 ) t.QPt() = 1.e-4 ;

		CADEBUG(printf("OUTPUT hits %d -> %d+%d = %d, QPt %f -> %f, SP %f, ok %d chi2 %f chi2ndf %f\n", nTrackHitsOld, nTrackHits, NTolerated, nTrackHits + NTolerated, ptOld, t.QPt(), t.SinPhi(), (int) ok, t.Chi2(), t.Chi2() / std::max(1,nTrackHits));)

		if (!ok && ++attempt < nAttempts)
		{
			for (int i = 0;i < track.NClusters();i++) clusters[track.FirstClusterRef() + i].fState &= AliHLTTPCGMMergedTrackHit::hwcfFlags;
			CADEBUG(printf("Track rejected, running refit\n");)
			continue;
		}

		track.SetOK(ok);
		track.SetNClustersFitted( nTrackHits );
		track.Param() = t;
		track.Alpha() = Alpha;
		break;
	}

	if (track.OK())
	{
	  int ind = track.FirstClusterRef();
	  const AliGPUCAParam &param = merger->SliceParam();
	  float alphaa = param.Alpha(clusters[ind].fSlice);
	  float xx = clusters[ind].fX;
	  float yy = clusters[ind].fY;
	  float zz = clusters[ind].fZ - track.Param().GetZOffset();
	  float sinA = CAMath::Sin( alphaa - track.Alpha());
	  float cosA = CAMath::Cos( alphaa - track.Alpha());
	  track.SetLastX( xx*cosA - yy*sinA );
	  track.SetLastY( xx*sinA + yy*cosA );
	  track.SetLastZ( zz );
	}
}

#ifdef GPUCA_GPUCODE

GPUg() void RefitTracks(AliHLTTPCGMMergedTrack* tracks, int nTracks, AliHLTTPCGMMergedTrackHit* clusters)
{
	for (int i = get_global_id(0);i < nTracks;i += get_global_size(0))
	{
		AliHLTTPCGMTrackParam::RefitTrack(tracks[i], i, &gGPUConstantMem.tpcMerger, clusters);
	}
}

#endif

GPUd() bool AliHLTTPCGMTrackParam::Rotate(float alpha)
{
	float cA = CAMath::Cos(alpha);
	float sA = CAMath::Sin(alpha);
	float x0 = fX;
	float sinPhi0 = fP[2], cosPhi0 = CAMath::Sqrt(1 - fP[2] * fP[2]);
	float cosPhi = cosPhi0 * cA + sinPhi0 * sA;
	float sinPhi = -cosPhi0 * sA + sinPhi0 * cA;
	float j0 = cosPhi0 / cosPhi;
	float j2 = cosPhi / cosPhi0;
	fX = x0 * cA + fP[0] * sA;
	fP[0] = -x0 * sA + fP[0] * cA;
	fP[2] = sinPhi + j2;
	fC[0] *= j0 * j0;
	fC[1] *= j0;
	fC[3] *= j0;
	fC[6] *= j0;
	fC[10] *= j0;

	fC[3] *= j2;
	fC[4] *= j2;
	fC[5] *= j2 * j2;
	fC[8] *= j2;
	fC[12] *= j2;
	if (cosPhi < 0)
	{ // change direction ( t0 direction is already changed in t0.UpdateValues(); )
		SinPhi() = -SinPhi();
		DzDs() = -DzDs();
		QPt() = -QPt();
		fC[3] = -fC[3];
		fC[4] = -fC[4];
		fC[6] = -fC[6];
		fC[7] = -fC[7];
		fC[10] = -fC[10];
		fC[11] = -fC[11];
	}
	return true;
}
