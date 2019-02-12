#ifndef ALIGPUPARAM_H
#define ALIGPUPARAM_H

#include "AliGPUCommonDef.h"
#include "AliGPUCommonMath.h"
#include "AliGPUTPCSettings.h"
#include "AliGPUTPCDef.h"
#include "AliGPUSettings.h"

struct AliGPUSettingsRec;
struct AliGPUSettingsEvent;

struct AliGPUParamSlice
{
	float Alpha;						// slice angle
	float CosAlpha, SinAlpha;			// sign and cosine of the slice angle
	float AngleMin, AngleMax;			// minimal and maximal angle
	float ZMin, ZMax;					// slice Z range
};

MEM_CLASS_PRE() class AliGPUParam
{
public:
	AliGPUSettingsRec rec;
	
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
	AliGPUParamSlice SliceParam[GPUCA_NSLICES];

#ifndef GPUCA_GPUCODE
	void SetDefaults(float solenoidBz);
	void SetDefaults(const AliGPUSettingsEvent* e, const AliGPUSettingsRec* r = NULL, const AliGPUSettingsDeviceProcessing* p = NULL);
	void UpdateEventSettings(const AliGPUSettingsEvent* e, const AliGPUSettingsDeviceProcessing* p = NULL);
	void LoadClusterErrors(bool Print = 0);
#endif
	
	GPUd() float Alpha( int iSlice ) const { if (iSlice >= GPUCA_NSLICES / 2) iSlice -= GPUCA_NSLICES / 2; if (iSlice >= GPUCA_NSLICES / 4) iSlice -= GPUCA_NSLICES / 2; return 0.174533f + DAlpha * iSlice;}
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
