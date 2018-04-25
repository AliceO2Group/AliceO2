#ifndef ALIHLTTPCCAMCINFO_H
#define ALIHLTTPCCAMCINFO_H

struct AliHLTTPCCAMCInfo
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
