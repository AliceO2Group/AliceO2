#ifndef ALIESDCALOTRIGGER_H
#define ALIESDCALOTRIGGER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include "AliVCaloTrigger.h"

class TArrayI;

/**
 * @class AliESDCaloTrigger
 * @brief Container with calorimeter trigger information in the ESD event
 *
 * @author  R. Guernane, LPSC Grenoble CNRS/IN2P3
 */
class AliESDCaloTrigger : public AliVCaloTrigger 
{
public:
	         AliESDCaloTrigger();
	         AliESDCaloTrigger(const AliESDCaloTrigger& ctrig);
	virtual ~AliESDCaloTrigger();
	
	AliESDCaloTrigger& operator=(const AliESDCaloTrigger& ctrig);
	
	Bool_t  IsEmpty() {return (fNEntries == 0);}

	virtual void Reset() {fCurrent = -1;}

	void    Allocate(Int_t size);
	void    DeAllocate(        ); 
	
        Bool_t  Add(Int_t col, Int_t row, Float_t amp, Float_t time, Int_t trgtimes[], Int_t ntrgtimes, Int_t trgts, Int_t trgbits);
        Bool_t  Add(Int_t col, Int_t row, Float_t amp, Float_t time, Int_t trgtimes[], Int_t ntrgtimes, Int_t trgts, Int_t subra, Int_t trgbits);
	
	void    SetL1Threshold(Int_t i, Int_t thr) {fL1Threshold[i] = thr;}
        void    SetL1Threshold(Int_t i, Int_t j, Int_t thr) {if (i) fL1DCALThreshold[j] = thr; else fL1Threshold[j] = thr;}
  
	void    SetL1V0(const Int_t* v) {for (int i = 0; i < 2; i++) fL1V0[i] = v[i];}
        void    SetL1V0(Int_t i, const Int_t* v) {
          if (i) {for (int j = 0; j < 2; j++) fL1DCALV0[j] = v[j];} else {for (int j = 0; j < 2; j++) fL1V0[j] = v[j];}
        }
 
        void    SetL1FrameMask(Int_t m) {fL1FrameMask = m;}
        void    SetL1FrameMask(Int_t i, Int_t m) {if (i) fL1DCALFrameMask = m; else fL1FrameMask = m;}
  
	void    SetTriggerBitWord(Int_t w) {fTriggerBitWord = w;}
	void    SetMedian(Int_t i, Int_t m) {fMedian[i] = m;}

	/**
	 * @brief Access to position of the current fastor channel
	 * @param[out] col Column of the current fastor in the detector plane
	 * @param[out] row Row of the current fastor in the detector plane
	 */
	void    GetPosition(       Int_t& col, Int_t& row           ) const;

	/**
	 * @brief Access to L0-amplitude of the current fastor channel
	 * @param[out] amp L0-amplitude for the given fastor channel
	 */
	void    GetAmplitude(      Float_t& amp                     ) const;

	void    GetTime(           Float_t& time                    ) const;
	
	/**
	 * @brief Get the trigger bits for a given fastor position
	 *
	 * Trigger bits define the starting position of online patches.
	 * They are defined in AliEMCALTriggerTypes.h. Note that for
	 * reconstructed patches an offset (MC offset) has to be taken
	 * into account
	 *
	 * @param[out] bits Trigger bits connected to a given position
	 */
	void    GetTriggerBits(    Int_t& bits                      ) const;

	/**
	 * @brief Get the number of L0 times for the current patch
	 *
	 * Level0 times are handled per L0 patch. Indexing is different
	 * with respect to the fastor indexing.
	 *
	 * @param[out] ntimes Number of level0
	 */
	void    GetNL0Times(       Int_t& ntimes                    ) const;

	/**
	 * @brief Get level0 times for the current L0 patch
	 * @param times L0 times for the current L0 patch
	 */
	void    GetL0Times(        Int_t  times[]                   ) const;

	/**
	 * @brief Get the number of entries in the trigger data
	 * @return Number of entries
	 */
	Int_t   GetEntries(                                         ) const {return fNEntries;}

    /**
     * @brief Get the L1 time sums (L1 ADC values) for the current fastor
     * @param[out] timesum L1 timesums for the current fastor
     */
        void    GetL1TimeSum(      Int_t& timesum                   ) const;

        /**
         * @brief Get the L1 time sums (L1 ADC values) for the current fastor
         * @return L1 timesums for the current fastor
         */
        Int_t   GetL1TimeSum(                                       ) const;
  
        void    GetL1SubRegion(    Int_t& subreg                    ) const;
        Int_t   GetL1SubRegion(                                     ) const;
  
	Int_t   GetL1Threshold(    Int_t  i                         ) const {return fL1Threshold[i];}
	Int_t   GetL1Threshold(    Int_t  i, Int_t j                ) const {return ((i)?fL1DCALThreshold[j]:fL1Threshold[j]);}
  
	Int_t   GetL1V0(           Int_t  i                         ) const {return fL1V0[i];}
        Int_t   GetL1V0(           Int_t  i, Int_t  j               ) const {return ((i)?fL1DCALV0[j]:fL1V0[j]);}
  
        Int_t   GetL1FrameMask(                                     ) const {return fL1FrameMask;}
        Int_t   GetL1FrameMask(    Int_t  i                         ) const {return ((i)?fL1DCALFrameMask:fL1FrameMask);}
  
        Int_t   GetMedian(         Int_t  i                         ) const {return (fMedian[i] & 0x3FFFF);}
  
        Int_t   GetTriggerBitWord(                                  ) const {return fTriggerBitWord;}
        void    GetTriggerBitWord( Int_t& bw                        ) const {bw = fTriggerBitWord;}

    /**
     * @brief Forward to next trigger entry (fastor / L0 patch)
     * @return True if successful (next entry existing), false otherwise (was already at the end of the buffer)
     */
	virtual Bool_t Next();

	virtual void Copy(TObject& obj) const;
	
	virtual void Print(const Option_t* opt) const;
	
private:

        Int_t    fNEntries;		///< Number of entries in the trigger object (usually mapped to fastor channels)
        Int_t    fCurrent;		///< Index of the current entry

    	/** Array of col positions for a trigger entry, one entry corresponds to a certain fastor channel */
	Int_t*   fColumn;             // [fNEntries]

	/** Array of row positions for a trigger entry, one entry corresponds to a certain fastor channel */
	Int_t*   fRow;                // [fNEntries]

	/** Array with L0 amplitudes for a trigger entry, one entry corresponds to a certain fastor channel */
	Float_t* fAmplitude;          // [fNEntries]

	/** Array of trigger times, one entry corresponds to a certain fastor channel */
	Float_t* fTime;               // [fNEntries]

	/** Array of the number of Level0 times, one entry corresponds to a certain L0 patch */
	Int_t*   fNL0Times;           // [fNEntries]

	/** Array of Level0 times, one entry corresponds to a certain L0 patch */
	TArrayI* fL0Times;            //

	/** Array of the L1 time sums (L1 ADC values), one entry corresponds to a certain fastor channel */
	Int_t*   fL1TimeSum;          // [fNEntries]

	/** Array of trigger bits: Each bit position corresponds to a certain trigger (L0/L1) fired. Note
	 * that there is a MC offset to be taken into account. Reconsturcted triggers start from the position
	 * of the MC offset. Trigger bits are defined in AliEMCALTriggerTypes.h. One entry correspons to a
	 * certain fastor position.
	 */
	Int_t*   fTriggerBits;        // [fNEntries]
	
	Int_t    fL1Threshold[4];     ///< L1 thresholds from raw data
	Int_t    fL1V0[2];            ///< L1 threshold components
	Int_t    fL1FrameMask;        ///< Validation flag for L1 data
  
        Int_t    fL1DCALThreshold[4]; ///< L1 thresholds from raw data
        Int_t*   fL1SubRegion;        // [fNEntries]
        Int_t    fL1DCALFrameMask;    ///< Validation flag for L1 data
        Int_t    fMedian[2];          ///< Background median
        Int_t    fTriggerBitWord;     ///< Trigger bit word
        Int_t    fL1DCALV0[2];        ///< L1 threshold components
	
    /// \cond CLASSIMP
	ClassDef(AliESDCaloTrigger, 8)
    /// \endcond
};
#endif

