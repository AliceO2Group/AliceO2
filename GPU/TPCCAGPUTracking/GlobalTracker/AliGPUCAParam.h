#ifndef ALIGPUCAPARAM_H
#define ALIGPUCAPARAM_H

#include "AliTPCCommonDef.h"
#include "AliTPCCommonMath.h"
#include "AliHLTTPCCASettings.h"
#include "AliHLTTPCCADef.h"

struct AliGPUCAParamSlice
{
	float Alpha;						// slice angle
	float CosAlpha, SinAlpha;			// sign and cosine of the slice angle
	float AngleMin, AngleMax;			// minimal and maximal angle
	float ZMin, ZMax;					// slice Z range
};

MEM_CLASS_PRE() class AliGPUCAParam
{
public:
	float DAlpha;						// angular size
	float RMin, RMax;					// slice R range
	float ErrX, ErrY, ErrZ;			// default cluster errors
	float PadPitch;						// pad pitch
	float BzkG;							// constant magnetic field value in kG
	float ConstBz;						// constant magnetic field value in kG*clight
	
	float HitPickUpFactor;				// multiplier for the chi2 window for hit pick up procedure
	int MaxTrackMatchDRow;				// maximal jump in TPC row for connecting track segments
	float NeighboursSearchArea;			// area in cm for the search of neighbours
	float TrackConnectionFactor;		// allowed distance in Chi^2/3.5 for neighbouring tracks
	float ClusterError2CorrectionY;		// correction for the squared cluster error during tracking
	float ClusterError2CorrectionZ;		// correction for the squared cluster error during tracking
	int MinNTrackClusters;				//* required min number of clusters on the track
	float MaxTrackQPt;					//* required max Q/Pt (==min Pt) of tracks
	char NWays;							//Do N fit passes in final fit of merger
	char NWaysOuter;					//Store outer param
	char AssumeConstantBz;				//Assume a constant magnetic field
	char ToyMCEventsFlag;				//events were build with home-made event generator
	char ContinuousTracking;			//Continuous tracking, estimate bz and errors for abs(z) = 125cm during seeding
	char RejectMode;					//0: no limit on rejection or missed hits, >0: break after n rejected hits, <0: reject at max -n hits
	float SearchWindowDZDR;				//Use DZDR window for seeding instead of vertex window
	float TrackReferenceX;				//Transport all tracks to this X after tracking (disabled if > 500)

	float RowX[HLTCA_ROW_COUNT];		// X-coordinate of rows
	AliGPUCAParamSlice SliceParam[36];

#ifndef HLTCA_GPUCODE
	void SetDefaults(float solenoidBz);
	void LoadClusterErrors(bool Print = 0);
#endif
	
	GPUd() float Alpha( int iSlice ) const { if (iSlice >= 18) iSlice -= 18; if (iSlice >= 9) iSlice -= 18; return 0.174533 + DAlpha * iSlice;}
	GPUd() void SetMinTrackPt( float v ){ MaxTrackQPt = CAMath::Abs(v)>0.001 ?1./CAMath::Abs(v) :1./0.001; }
	GPUd() float GetClusterRMS( int yz, int type, float z, float angle2 ) const;
	GPUd() void GetClusterRMS2( int row, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const;

	GPUd() float GetClusterError2( int yz, int type, float z, float angle2 ) const;
	GPUd() void GetClusterErrors2( int row, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const;
	
	void Slice2Global( int iSlice, float x, float y, float z, float *X, float *Y, float *Z ) const;
	void Global2Slice( int iSlice, float x, float y, float z, float *X, float *Y, float *Z ) const;

protected:
	float ParamRMS0[2][3][4];			// cluster shape parameterization coeficients
	float ParamS0Par[2][3][6];			// cluster error parameterization coeficients
};

#endif
