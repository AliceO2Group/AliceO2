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
//
// Container for the reference distributions for the TRD PID
// Provides storing of the references together with the mometum steps
// they were taken
// More information can be found in the implementation file
//
#ifndef ALITRDPIDREFERENCE_H
#define ALITRDPIDREFERENCE_H

#include <TNamed.h>
#include <TArrayF.h>
#include "AliPID.h"

class TObjArray;

class AliTRDPIDReference : public TNamed{
public:
	AliTRDPIDReference();
	AliTRDPIDReference(const Char_t *name);
    AliTRDPIDReference(const AliTRDPIDReference &ref, Int_t NofCharges=1);
	AliTRDPIDReference &operator=(const AliTRDPIDReference &ref);
	~AliTRDPIDReference();

    void SetNumberOfMomentumBins(Int_t nBins, Float_t *momenta, Int_t NofCharges=1);
    void AddReference(TObject *histo, AliPID::EParticleType spec, Int_t pbin, Int_t Charge=0);

	// Derive reference
    TObject *GetLowerReference(AliPID::EParticleType spec, Float_t p, Float_t &pLower, Int_t Charge=0) const;
    TObject *GetUpperReference(AliPID::EParticleType spec, Float_t p, Float_t &pUpper, Int_t Charge=0) const;

	Int_t GetNumberOfMomentumBins() const { return fMomentumBins.GetSize(); }
	void Print(const Option_t *) const;
private:
	enum{
		kIsOwner = BIT(14)
	};
	TObjArray *fRefContainer;     // Histogram container
	TArrayF fMomentumBins;        // Momentum Bins

	ClassDef(AliTRDPIDReference, 1)		// Container for TRD references
};
#endif

