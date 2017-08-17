#ifndef ALIHLTTPCCAMCINFO_H
#define ALIHLTTPCCAMCINFO_H

struct AliHLTTPCCAMCInfo
{
	int fCharge;
	bool fPrim;
	bool fPrimDaughters;
	int fPID;
	float fX;
	float fY;
	float fZ;
	float fPx;
	float fPy;
	float fPz;
	float fNWeightCls;
};

#endif
