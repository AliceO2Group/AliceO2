#ifndef ALIGPUCAPARAM_H
#define ALIGPUCAPARAM_H

#include "AliTPCCommonDef.h"
#include "AliTPCCommonMath.h"
#include "AliHLTTPCCASettings.h"
#include "AliHLTTPCCADef.h"
#include "AliGPUCASettings.h"

struct AliGPUCASettingsRec;
struct AliGPUCASettingsEvent;

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
	AliGPUCASettingsRec rec;
	
	float DAlpha;						// angular size
	float RMin, RMax;					// slice R range
	float ErrX, ErrY, ErrZ;				// default cluster errors
	float PadPitch;						// pad pitch
	float BzkG;							// constant magnetic field value in kG
	float ConstBz;						// constant magnetic field value in kG*clight
	
	char AssumeConstantBz;				//Assume a constant magnetic field
	char ToyMCEventsFlag;				//events were build with home-made event generator
	char ContinuousTracking;			//Continuous tracking, estimate bz and errors for abs(z) = 125cm during seeding
	char resetTimers;					//Reset benchmark timers before event processing
	int debugLevel;						//Debug level
	int continuousMaxTimeBin;			//Max time bin for continuous tracking
	float RowX[GPUCA_ROW_COUNT];		// X-coordinate of rows
	AliGPUCAParamSlice SliceParam[36];

#ifndef GPUCA_GPUCODE
	void SetDefaults(float solenoidBz);
	void SetDefaults(const AliGPUCASettingsEvent* e, const AliGPUCASettingsRec* r = NULL, const AliGPUCASettingsDeviceProcessing* p = NULL);
	void UpdateEventSettings(const AliGPUCASettingsEvent* e, const AliGPUCASettingsDeviceProcessing* p = NULL);
	void LoadClusterErrors(bool Print = 0);
#endif
	
	GPUd() float Alpha( int iSlice ) const { if (iSlice >= 18) iSlice -= 18; if (iSlice >= 9) iSlice -= 18; return 0.174533 + DAlpha * iSlice;}
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
