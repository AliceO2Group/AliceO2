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

#define HLTCA_CADEBUG 0

#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCGMPhysicalTrackModel.h"
#include "AliHLTTPCGMPropagator.h"
#include "AliHLTTPCGMBorderTrack.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCGMPolynomialField.h"
#ifndef HLTCA_STANDALONE
#include "AliExternalTrackParam.h"
#endif
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCAClusterErrorStat.h"

#include <cmath>
#include <stdlib.h>

GPUd() bool AliHLTTPCGMTrackParam::Fit(const AliHLTTPCGMPolynomialField* field, AliHLTTPCGMMergedTrackHit* clusters, const AliHLTTPCCAParam &param, int &N, float &Alpha, int attempt, float maxSinPhi)
{
  const float kRho = 1.025e-3;//0.9e-3;
  const float kRadLen = 29.532;//28.94;
  const float kDeg2Rad = 3.1415926535897 / 180.f;
  const float kSectAngle = 2*3.1415926535897 / 18.f;
  
  CADEBUG(static int nTracks = 0;nTracks++;)
  AliHLTTPCCAClusterErrorStat errorStat(N);

  AliHLTTPCGMPropagator prop;
  prop.SetMaterial( kRadLen, kRho );
  prop.SetPolynomialField( field );  
  prop.SetContinuousTracking( param.GetContinuousTracking() );
  prop.SetMaxSinPhi( maxSinPhi );
  prop.SetToyMCEventsFlag( param.ToyMCEventsFlag());

  if (param.GetContinuousTracking())
  {  
    fP[1] += fZOffset;
    
    const float cosPhi = AliHLTTPCCAMath::Sqrt(1 - fP[2] * fP[2]);
    const float dxf = -AliHLTTPCCAMath::Abs(fP[2]);
    const float dyf = cosPhi * (fP[2] > 0 ? 1. : -1.);
    const float r = 1./fabs(fP[4] * field->GetNominalBz());
    float xp = fX + dxf * r;
    float yp = fP[0] + dyf * r;
    //printf("\nX %f Y %f SinPhi %f QPt %f R %f --> XP %f YP %f\n", fX, fP[0], fP[2], fP[4], r, xp, yp);
    const float r2 = (r + AliHLTTPCCAMath::Sqrt(xp * xp + yp * yp)) / 2.; //Improve the radius by taking into acount both points we know (00 and xy).
    xp = fX + dxf * r2;
    yp = fP[0] + dyf * r2;
    //printf("X %f Y %f SinPhi %f QPt %f R %f --> XP %f YP %f\n", fX, fP[0], fP[2], fP[4], r2, xp, yp);
    float atana = AliHLTTPCCAMath::ATan2(CAMath::Abs(xp), CAMath::Abs(yp));
    float atanb = AliHLTTPCCAMath::ATan2(CAMath::Abs(fX - xp), CAMath::Abs(fP[0] - yp));
    //printf("Tan %f %f (%f %f)\n", atana, atanb, fX - xp, fP[0] - yp);
    const float dS = (xp > 0 ? (atana + atanb) : (atanb - atana)) * r;
    float dz = dS * fP[3];
    //printf("Track Z %f, Z0 %f (dS %f, dZds %f)             - Direction %f to %f: %f\n", fP[1], dz, dS, fP[3], clusters[0].fZ, clusters[N - 1].fZ, clusters[0].fZ - clusters[N - 1].fZ);
    if (CAMath::Abs(dz) > 250.) dz = dz > 0 ? 250. : -250.;
    if (fP[1] * (fP[1] - dz) < 0)
    {
      fZOffset = clusters[N - 1].fZ;
    }
    else
    {
      fZOffset = fP[1] - dz; 
    }
    fP[1] -= fZOffset;
  }

  int nWays = param.GetNWays();
  int maxN = N;
  int ihitStart = 0;
  float covYYUpd = 0.;
  float lastUpdateX = -1.;
  
  ConstrainSinPhi();
  for (int iWay = 0;iWay < nWays;iWay++)
  {
    if (iWay && param.GetNWaysOuter() && iWay == nWays - 1)
    {
        for (int i = 0;i < 5;i++) fOuterParam.fP[i] = fP[i];
        fOuterParam.fP[1] += fZOffset;
        for (int i = 0;i < 15;i++) fOuterParam.fC[i] = fC[i];
        fOuterParam.fX = fX;
        fOuterParam.fAlpha = prop.GetAlpha();
    }

    int resetT0 = CAMath::Max(10.f, CAMath::Min(40.f, 150.f / fP[4]));
    const bool rejectChi2ThisRound = ( nWays == 1 || iWay >= 1 );
    const double kDeg2Rad = 3.14159265358979323846/180.;
    const float maxSinForUpdate = CAMath::Sin(70.*kDeg2Rad);
    if (rejectChi2ThisRound && attempt == 0) prop.SetSpecialErrors(true);
  
    ResetCovariance();
    prop.SetFitInProjections(iWay != 0);
    prop.SetTrack( this, iWay ? prop.GetAlpha() : Alpha);
    CADEBUG(printf("Fitting track %d way %d (sector %d, alpha %f)\n", nTracks, iWay, (int) (prop.GetAlpha() / kSectAngle + 0.5) + (fP[1] < 0 ? 18 : 0), prop.GetAlpha());)

    N = 0;
    lastUpdateX = -1;
    const bool inFlyDirection = iWay & 1;
    unsigned char lastLeg = clusters[ihitStart].fLeg;
    const int wayDirection = (iWay & 1) ? -1 : 1;
    int ihit = ihitStart;
    for(;ihit >= 0 && ihit<maxN;ihit += wayDirection)
    {
      if (clusters[ihit].fState & AliHLTTPCGMMergedTrackHit::flagReject) continue; // hit is excluded from fit
      CADEBUG(printf("\tHit %3d/%3d Row %3d: Cluster Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f\n", ihit, maxN, clusters[ihit].fRow, param.Alpha(clusters[ihit].fSlice), clusters[ihit].fX, clusters[ihit].fY, clusters[ihit].fZ);)
      
      float xx = clusters[ihit].fX;
      float yy = clusters[ihit].fY;
      float zz = clusters[ihit].fZ - fZOffset;
      unsigned char clusterState = clusters[ihit].fState;
      int ihitMergeFirst = ihit;
      if (ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].fRow == clusters[ihit + wayDirection].fRow)
      {
          float count = 0.f;
          xx = yy = zz = 0.f;
          clusterState = 0;
          float maxDistY, maxDistZ;
          prop.GetErr2(maxDistY, maxDistZ, param, zz, clusters[ihit].fRow, 0);
          while(true)
          {
              float dy = clusters[ihit].fY - fP[0], dz = clusters[ihit].fZ - fP[1];
              if (dy * dy > 9.f * maxDistY || dz * dz > 9.f * maxDistZ)
              {
                clusters[ihit].fState |= AliHLTTPCGMMergedTrackHit::flagRejectDistance;
              }
              else
              {
                CADEBUG(printf("\t\tMerging hit row %d X %f Y %f Z %f\n", clusters[ihit].fRow, clusters[ihit].fX, clusters[ihit].fY, clusters[ihit].fZ);)
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
          if (count < 0.1) continue;
          xx /= count;
          yy /= count;
          zz /= count;
          CADEBUG(printf("\t\tDouble row (%f tot charge, %d -> %d)\n", count, ihitMergeFirst, ihit);)
      }
      
      bool changeDirection = (clusters[ihit].fLeg - lastLeg) & 1;
      CADEBUG(if(changeDirection) printf("\t\tChange direction\n");)
      CADEBUG(printf("\tLeg %3d%14sTrack   Alpha %8.3f %s, X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SinPhi %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f SPPt %8.3f YSP %8.3f\n", (int) clusters[ihit].fLeg, "", prop.GetAlpha(), (fabs(prop.GetAlpha() - param.Alpha(clusters[ihit].fSlice)) < 0.01 ? "   " : " R!"), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), "", sqrt(fC[0]), sqrt(fC[2]), sqrt(fC[5]), sqrt(fC[14]), fC[10], fC[12], fC[3]);)
      int err = prop.PropagateToXAlpha(xx, param.Alpha(clusters[ihit].fSlice), inFlyDirection );
      if (err == -2) //Rotation failed, try to bring to new x with old alpha first, rotate, and then propagate to x, alpha
      {
          CADEBUG(printf("REROTATE\n");)
          if (prop.PropagateToXAlpha(xx, prop.GetAlpha(), inFlyDirection ) == 0)
            err = prop.PropagateToXAlpha(xx, param.Alpha(clusters[ihit].fSlice), inFlyDirection );
      }
      
      CADEBUG(printf("\t%21sPropaga Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SinPhi %5.2f (%5.2f)   ---   Res %8.3f %8.3f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f SPPt %8.3f YSP %8.3f   -   Err %d", "", prop.GetAlpha(), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), fP[0] - yy, fP[1] - zz, sqrt(fC[0]), sqrt(fC[2]), sqrt(fC[5]), sqrt(fC[14]), fC[10], fC[12], fC[3], err);)

      if (err == 0 && changeDirection)
      {
          const float mirrordY = prop.GetMirroredYTrack();
          CADEBUG(printf(" -- MiroredY: %f --> %f", fP[0], mirrordY);)
          if (fabs(yy - fP[0]) > fabs(yy - mirrordY))
          {
              CADEBUG(printf(" - Mirroring!!!");)
              prop.Mirror(inFlyDirection);
              float err2Y, err2Z;
              prop.GetErr2(err2Y, err2Z, param, zz, clusters[ihit].fRow, clusterState);
              prop.Model().Y() = fP[0] = yy;
              prop.Model().Z() = fP[1] = zz;
              lastUpdateX = fX;
              if (fC[0] < err2Y) fC[0] = err2Y;
              if (fC[2] < err2Z) fC[2] = err2Z;
              if (fabs(fC[5]) < 0.1) fC[5] = fC[5] > 0 ? 0.1 : -0.1;
              if (fC[9] < 1.) fC[9] = 1.;
              prop.SetTrack(this, prop.GetAlpha());

              fNDF = -3;
              fChi2 = 0;
              lastLeg = clusters[ihit].fLeg;
              N++;
              resetT0 = CAMath::Max(10.f, CAMath::Min(40.f, 150.f / fP[4]));
              CADEBUG(printf("\n");)
              CADEBUG(printf("\t%21sMirror  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SinPhi %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f SPPt %8.3f YSP %8.3f\n", "", prop.GetAlpha(), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), "", sqrt(fC[0]), sqrt(fC[2]), sqrt(fC[5]), sqrt(fC[14]), fC[10], fC[12], fC[3]);)
              continue;
          }
      }

      const int err2 = fNDF > 0 && CAMath::Abs(prop.GetSinPhi0())>=maxSinForUpdate;
      if ( err || err2 )
      {
        if (rejectChi2ThisRound)
        {
          if (fNDF > 0 && (fabs(yy - fP[0]) > 3 || fabs(zz - fP[1]) > 3)) MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, AliHLTTPCGMMergedTrackHit::flagRejectDistance);
          else if (err && err >= -3) MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, AliHLTTPCGMMergedTrackHit::flagRejectErr);
        }
        
        CADEBUG(printf(" --- break (%d, %d)\n", err, err2);)
        continue;
      }
      CADEBUG(printf("\n");)
      errorStat.Fill(xx, yy, zz, prop.GetAlpha(), fX, fP, fC, ihit, iWay);
      
      int retVal;
      float threshold = 3. + (lastUpdateX >= 0 ? (fabs(fX - lastUpdateX) / 2) : 0.);
      if (fNDF > 5 && (fabs(yy - fP[0]) > threshold || fabs(zz - fP[1]) > threshold)) retVal = 2;
      else retVal = prop.Update( yy, zz, clusters[ihit].fRow, param, clusterState, rejectChi2ThisRound);
      CADEBUG(printf("\t%21sFit     Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SinPhi %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f SPPt %8.3f YSP %8.3f   -   Err %d\n", "", prop.GetAlpha(), fX, fP[0], fP[1], fP[4], prop.GetQPt0(), fP[2], prop.GetSinPhi0(), "", sqrt(fC[0]), sqrt(fC[2]), sqrt(fC[5]), sqrt(fC[14]), fC[10], fC[12], fC[3], retVal);)

      if (retVal == 0) // track is updated
      {
        lastUpdateX = fX;
        covYYUpd = fC[0];
        ihitStart = ihit;
        N++;
        float dy = fP[0] - prop.Model().Y();
        float dz = fP[1] - prop.Model().Z();
        if (AliHLTTPCCAMath::Abs(fP[4]) > 10 && --resetT0 <= 0 && AliHLTTPCCAMath::Abs(fP[2]) < 0.15 && dy*dy+dz*dz>1)
        {
            CADEBUG(printf("Reinit linearization\n");)
            prop.SetTrack(this, prop.GetAlpha());
        }
      }
      else if (retVal == 2) // cluster far away form the track
      {
        if (rejectChi2ThisRound) MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, AliHLTTPCGMMergedTrackHit::flagRejectDistance);
      }
      else break; // bad chi2 for the whole track, stop the fit
    }
    ConstrainSinPhi();
  }
  
  bool ok = N >= TRACKLET_SELECTOR_MIN_HITS(fP[4]) && CheckNumericalQuality(covYYUpd);
  if (!ok) return(false);
  
  Alpha = prop.GetAlpha();
  if (param.GetTrackReferenceX() <= 500)
  {
      for (int k = 0;k < 3;k++) //max 3 attempts
      {
          int err = prop.PropagateToXAlpha(param.GetTrackReferenceX(), Alpha, 0);
          ConstrainSinPhi();
          if (fabs(fP[0]) <= fX * tan(kSectAngle / 2.f)) break;
          float dAngle = floor(atan2(fP[0], fX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
          Alpha += dAngle;
          if (err || k == 2)
          {
              Rotate(dAngle);
              break;
          }
      }
      
  }
  else if (fabs(fP[0]) > fX * tan(kSectAngle / 2.f))
  {
      float dAngle = floor(atan2(fP[0], fX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
      Rotate(dAngle);
      Alpha += dAngle;
  }
  if (Alpha > 3.1415926535897) Alpha -= 2*3.1415926535897;
  else if (Alpha <= -3.1415926535897) Alpha += 2*3.1415926535897;
  
  return(ok);
}

GPUd() bool AliHLTTPCGMTrackParam::CheckNumericalQuality(float overrideCovYY) const
{
  //* Check that the track parameters and covariance matrix are reasonable
  bool ok = AliHLTTPCCAMath::Finite(fX) && AliHLTTPCCAMath::Finite( fChi2 );
  CADEBUG(printf("OK %d - ", (int) ok); for (int i = 0;i < 5;i++) printf("%f ", fP[i]); printf(" - "); for (int i = 0;i < 15;i++) printf("%f ", fC[i]); printf("\n");)
  const float *c = fC;
  for ( int i = 0; i < 15; i++ ) ok = ok && AliHLTTPCCAMath::Finite( c[i] );
  CADEBUG(printf("OK1 %d\n", (int) ok);)
  for ( int i = 0; i < 5; i++ ) ok = ok && AliHLTTPCCAMath::Finite( fP[i] );
  CADEBUG(printf("OK2 %d\n", (int) ok);)
  if ( c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0 ) ok = 0;
  if ( (overrideCovYY > 0 ? overrideCovYY : c[0]) > 4.*4. || c[2] > 4.*4. || c[5] > 2.*2. || c[9] > 2.*2. ) ok = 0;
  CADEBUG(printf("OK3 %d\n", (int) ok);)
  if ( fabs( fP[2] ) > HLTCA_MAX_SIN_PHI ) ok = 0;
  CADEBUG(printf("OK4 %d\n", (int) ok);)
  if( ok ){
    ok = ok 
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
  }
  CADEBUG(printf("OK5 %d\n", (int) ok);)
  return ok;
}

#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)
bool AliHLTTPCGMTrackParam::GetExtParam( AliExternalTrackParam &T, double alpha ) const
{
  //* Convert from AliHLTTPCGMTrackParam to AliExternalTrackParam parameterisation,
  //* the angle alpha is the global angle of the local X axis

  bool ok = CheckNumericalQuality();

  double par[5], cov[15];
  for ( int i = 0; i < 5; i++ ) par[i] = fP[i];
  for ( int i = 0; i < 15; i++ ) cov[i] = fC[i];

  if ( par[2] > HLTCA_MAX_SIN_PHI ) par[2] = HLTCA_MAX_SIN_PHI;
  if ( par[2] < -HLTCA_MAX_SIN_PHI ) par[2] = -HLTCA_MAX_SIN_PHI;

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
  if ( fP[2] > HLTCA_MAX_SIN_PHI ) fP[2] = HLTCA_MAX_SIN_PHI;
  if ( fP[2] < -HLTCA_MAX_SIN_PHI ) fP[2] = -HLTCA_MAX_SIN_PHI;
}
#endif

GPUd() void AliHLTTPCGMTrackParam::RefitTrack(AliHLTTPCGMMergedTrack &track, const AliHLTTPCGMPolynomialField* field, AliHLTTPCGMMergedTrackHit* clusters, const AliHLTTPCCAParam& param)
{
	if( !track.OK() ) return;    

	const int nAttempts = 2;
	for (int attempt = 0;;)
	{
		int nTrackHits = track.NClusters();
		AliHLTTPCGMTrackParam t = track.Param();
		float Alpha = track.Alpha();  
		CADEBUG(int nTrackHitsOld = nTrackHits; float ptOld = t.QPt();)
		bool ok = t.Fit( field, clusters + track.FirstClusterRef(), param, nTrackHits, Alpha, attempt );
		
		if ( fabs( t.QPt() ) < 1.e-4 ) t.QPt() = 1.e-4 ;

		CADEBUG(printf("OUTPUT hits %d -> %d, QPt %f -> %f, SinPhi %f, ok %d chi2 %f chi2ndf %f\n", nTrackHitsOld, nTrackHits, ptOld, t.QPt(), t.SinPhi(), (int) ok, t.Chi2(), t.Chi2() / std::max(1,nTrackHits));)
		
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

	{
	  int ind = track.FirstClusterRef();
	  float alphaa = param.Alpha(clusters[ind].fSlice);
	  float xx = clusters[ind].fX;
	  float yy = clusters[ind].fY;
	  float zz = clusters[ind].fZ - track.Param().GetZOffset();
	  float sinA = AliHLTTPCCAMath::Sin( alphaa - track.Alpha());
	  float cosA = AliHLTTPCCAMath::Cos( alphaa - track.Alpha());
	  track.SetLastX( xx*cosA - yy*sinA );
	  track.SetLastY( xx*sinA + yy*cosA );
	  track.SetLastZ( zz );
	}
}

#ifdef HLTCA_GPUCODE

GPUg() void RefitTracks(AliHLTTPCGMMergedTrack* tracks, int nTracks, const AliHLTTPCGMPolynomialField* field, AliHLTTPCGMMergedTrackHit* clusters, AliHLTTPCCAParam* param)
{
	for (int i = get_global_id(0);i < nTracks;i += get_global_size(0))
	{
	  AliHLTTPCGMTrackParam::RefitTrack(tracks[i], field, clusters, *param);
	}
}

#endif

GPUd() bool AliHLTTPCGMTrackParam::Rotate( float alpha )
{
    float cA = CAMath::Cos( alpha );
    float sA = CAMath::Sin( alpha );
    float x0 = fX;
    float sinPhi0 = fP[2], cosPhi0 = CAMath::Sqrt(1 - fP[2] * fP[2]);
    float cosPhi =  cosPhi0 * cA + sinPhi0 * sA;
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
    if( cosPhi <0 ){ // change direction ( t0 direction is already changed in t0.UpdateValues(); )
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
