#ifndef GPUTPCMCINFO_H
#define GPUTPCMCINFO_H

struct GPUTPCMCInfo
{
	int fCharge;
	char fPrim;
	char fPrimDaughters;
	int fPID;
	float fX;
	float fY;
	float fZ;
	float fPx;
	float fPy;
	float fPz;
	float fGenRadius;
};

#endif
