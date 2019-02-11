/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/*
 

 

Author: R. GUERNANE LPSC Grenoble CNRS/IN2P3
*/

#include "AliESDCaloTrigger.h"
#include "AliLog.h"

#include "TArrayI.h"
#include "Riostream.h"
#include <cstdlib>

ClassImp(AliESDCaloTrigger)

//_______________
AliESDCaloTrigger::AliESDCaloTrigger() : AliVCaloTrigger(),
fNEntries(0),
fCurrent(-1),
fColumn(0x0),
fRow(0x0),
fAmplitude(0x0),
fTime(0x0),
fNL0Times(0x0),
fL0Times(new TArrayI()),
fL1TimeSum(0x0),
fTriggerBits(0x0),
fL1Threshold(),
fL1V0(),
fL1FrameMask(0),
fL1DCALThreshold(),
fL1SubRegion(0x0),
fL1DCALFrameMask(0),
fMedian(),
fTriggerBitWord(0),
fL1DCALV0()
{
	//
	for (int i = 0; i < 4; i++) {fL1Threshold[i] = fL1DCALThreshold[i] = 0;}
	fL1V0[0] = fL1V0[1] = 0;
        fL1DCALV0[0] = fL1DCALV0[1] = 0;
	fMedian[0] = fMedian[1] = 0;
}

//_______________
AliESDCaloTrigger::AliESDCaloTrigger(const AliESDCaloTrigger& src) : AliVCaloTrigger(src),
fNEntries(0),
fCurrent(-1),
fColumn(0x0),
fRow(0x0),
fAmplitude(0x0),
fTime(0x0),
fNL0Times(0x0),
fL0Times(new TArrayI()),
fL1TimeSum(0x0),
fTriggerBits(0x0),
fL1Threshold(),
fL1V0(),
fL1FrameMask(0),
fL1DCALThreshold(),
fL1SubRegion(0x0),
fL1DCALFrameMask(0),
fMedian(),
fTriggerBitWord(0),
fL1DCALV0()
{
	//
	src.Copy(*this);
}

//_______________
AliESDCaloTrigger::~AliESDCaloTrigger()
{
	//
	if (fNEntries) DeAllocate();
	
	delete fL0Times; fL0Times = 0x0;
}

//_______________
void AliESDCaloTrigger::DeAllocate()
{
	//
	delete [] fColumn;      fColumn      = 0x0;
	delete [] fRow;         fRow         = 0x0;     
	delete [] fAmplitude;   fAmplitude   = 0x0;
	delete [] fTime;        fTime        = 0x0;   
	delete [] fNL0Times;    fNL0Times    = 0x0;
	delete [] fL1TimeSum;   fL1TimeSum   = 0x0;
        delete [] fL1SubRegion; fL1SubRegion = 0x0;
	delete [] fTriggerBits; fTriggerBits = 0x0;

	fNEntries =  0;
	fCurrent  = -1;

	fL0Times->Reset();
}

//_______________
AliESDCaloTrigger& AliESDCaloTrigger::operator=(const AliESDCaloTrigger& src)
{
	//
	if (this != &src) src.Copy(*this);
	
	return *this;
}

//_______________
void AliESDCaloTrigger::Copy(TObject &obj) const 
{	
	//
	AliVCaloTrigger::Copy(obj);
	
	AliESDCaloTrigger& dest = static_cast<AliESDCaloTrigger&>(obj);

	if (dest.fNEntries) dest.DeAllocate();
	
	dest.Allocate(fNEntries);

	Bool_t newclass=1;
	if (fL1SubRegion==0)
	  newclass=0;
	for (Int_t i = 0; i < fNEntries; i++) 
        {
           Int_t times[10];
           for (Int_t j = 0; j < 10; j++) times[j] = fL0Times->At(10 * i + j);
	   Int_t l1subreg = 0;
	   if (newclass)
	     l1subreg=fL1SubRegion[i];
           dest.Add(fColumn[i], fRow[i], fAmplitude[i], fTime[i], times, fNL0Times[i], fL1TimeSum[i], l1subreg, fTriggerBits[i]);
        }	

        for (int i = 0; i < 4; i++) dest.SetL1Threshold(i, fL1Threshold[i]);
        for (int i = 0; i < 4; i++) dest.SetL1Threshold(1, i, fL1DCALThreshold[i]);
	
        dest.SetL1V0(fL1V0);
        dest.SetL1V0(1, fL1DCALV0);
        dest.SetL1FrameMask(fL1FrameMask);
        dest.SetL1FrameMask(1, fL1DCALFrameMask);
}

//_______________
void AliESDCaloTrigger::Allocate(Int_t size)
{
	//
	if (!size) return;
	
	fNEntries = size;
	
	fColumn      = new   Int_t[fNEntries];
	fRow         = new   Int_t[fNEntries];
	fAmplitude   = new Float_t[fNEntries];
	fTime        = new Float_t[fNEntries];
	fNL0Times    = new   Int_t[fNEntries];
	fL1TimeSum   = new   Int_t[fNEntries];
        fL1SubRegion = new   Int_t[fNEntries];
	fTriggerBits = new   Int_t[fNEntries];

	for (Int_t i = 0; i < fNEntries; i++) 
	{
	  fColumn[i]      = 0;
	  fRow[i]         = 0;
	  fAmplitude[i]   = 0;
	  fTime[i]        = 0;
	  fNL0Times[i]    = 0;
	  fL1TimeSum[i]   = 0;
          fL1SubRegion[i] = 0;
	  fTriggerBits[i] = 0;
	}
	
	fL0Times->Set(fNEntries * 10);
}

//_______________
Bool_t AliESDCaloTrigger::Add(Int_t col, Int_t row, Float_t amp, Float_t time, Int_t trgtimes[], Int_t ntrgtimes, Int_t trgts, Int_t trgbits)
{
	//
	fCurrent++;
	
	     fColumn[fCurrent] = col;
	        fRow[fCurrent] = row;
	  fAmplitude[fCurrent] = amp;
	       fTime[fCurrent] = time;
	   fNL0Times[fCurrent] = ntrgtimes;
	  fL1TimeSum[fCurrent] = trgts;	
	fTriggerBits[fCurrent] = trgbits;
	
	if (ntrgtimes > 9) 
	{
		AliError("Should not have more than 10 L0 times");
		return kFALSE;
	}
	
	for (Int_t i = 0; i < fNL0Times[fCurrent]; i++) fL0Times->AddAt(trgtimes[i], 10 * fCurrent + i);

	return kTRUE;
}

//_______________
Bool_t AliESDCaloTrigger::Add(Int_t col, Int_t row, Float_t amp, Float_t time, Int_t trgtimes[], Int_t ntrgtimes, Int_t trgts, Int_t subra, Int_t trgbits)
{
        //
        Add(col, row, amp, time, trgtimes, ntrgtimes, trgts, trgbits);
        fL1SubRegion[fCurrent] = subra; 

        return kTRUE;
}

//_______________
Bool_t AliESDCaloTrigger::Next()
{
	//
	if (fCurrent >= fNEntries - 1 || !fNEntries) return kFALSE;
	
	fCurrent++;
	
	return kTRUE;
}

//_______________
void AliESDCaloTrigger::GetPosition(Int_t& col, Int_t& row) const
{
	//
	if (fCurrent == -1) return;
	
	col = fColumn?fColumn[fCurrent]:0;
	row =    fRow?fRow[fCurrent]:0;
}

//_______________
void AliESDCaloTrigger::GetAmplitude(Float_t& amp) const
{
	//
	if (fCurrent == -1) return;

	amp = fAmplitude?fAmplitude[fCurrent]:0;
}

//_______________
void AliESDCaloTrigger::GetTime(Float_t& time) const
{
	//
	if (fCurrent == -1) return;

	time = fTime?fTime[fCurrent]:0;
}

//_______________
void AliESDCaloTrigger::GetL1TimeSum(Int_t& amp) const
{
	//	
	if (fCurrent == -1) return;

	amp = fL1TimeSum?fL1TimeSum[fCurrent]:0;
}

//_______________
Int_t AliESDCaloTrigger::GetL1TimeSum() const
{
        //      
        if (fCurrent == -1) return -1;

        return ((fL1TimeSum)?fL1TimeSum[fCurrent]:0);
}

//_______________
void AliESDCaloTrigger::GetL1SubRegion(Int_t& sb) const
{
        //      
        if (fCurrent == -1) return;

        sb = fL1SubRegion?fL1SubRegion[fCurrent]:0;
}

//_______________
Int_t AliESDCaloTrigger::GetL1SubRegion() const
{
        //      
        if (fCurrent == -1) return -1;

        return ((fL1SubRegion)?fL1SubRegion[fCurrent]:0);
}

//_______________
void AliESDCaloTrigger::GetNL0Times(Int_t& ntimes) const
{
	//
	if (fCurrent == -1) return;

	ntimes = fNL0Times?fNL0Times[fCurrent]:0;
}

//_______________
void AliESDCaloTrigger::GetTriggerBits(Int_t& bits) const
{
	//
	if (fCurrent == -1) return;

	bits = fTriggerBits?fTriggerBits[fCurrent]:0;
}

//_______________
void AliESDCaloTrigger::GetL0Times(Int_t times[]) const
{
	//
	if (fCurrent == -1) return;
	
	if (fNL0Times && fL0Times) {
	  for (Int_t i = 0; i < fNL0Times[fCurrent]; i++) times[i] = fL0Times->At(10 * fCurrent + i);
	}
}

//_______________
void AliESDCaloTrigger::Print(const Option_t* /*opt*/) const
{
	//
	if (fCurrent == -1) return;
	if (!fColumn) return;
	if (!fRow) return;
	if (!fAmplitude) return;
	if (!fTime) return;
	if (!fNL0Times) return;
	if (!fL0Times) return;
	if (!fL1TimeSum) return;
	if (!fTriggerBits) return;

	printf("============\n");
	printf("--L0:\n");
	printf("\tPOSITION (X: %2d Y: %2d) / FITTED F-ALTRO (AMP: %4f TIME: %3f)\n", 
		   fColumn[fCurrent], fRow[fCurrent], fAmplitude[fCurrent], fTime[fCurrent]);
	printf("\t%d L0 TIMES (", fNL0Times[fCurrent]); 
	for (Int_t i = 0; i < fNL0Times[fCurrent]; i++) printf("%2d ",fL0Times->At(10 * fCurrent + i));
	printf(")\n");
	printf("--L1:\n");
	printf("\tTIME SUM: %4d\n", fL1TimeSum[fCurrent]);
	printf("\tHIGH THRESHOLDS (GAMMA: %4d, JET: %4d)\n", fL1Threshold[0], fL1Threshold[1]);
	printf("\tLOW THRESHOLDS (GAMMA: %4d, JET: %4d)\n", fL1Threshold[2], fL1Threshold[3]);
	printf("--TRIGGER BITS: 0x%x\n", fTriggerBits[fCurrent]);
}	
