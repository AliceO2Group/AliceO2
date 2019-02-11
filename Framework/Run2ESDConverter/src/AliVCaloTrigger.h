#ifndef ALIVCALOTRIGGER_H
#define ALIVCALOTRIGGER_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include <TNamed.h>

/**
 * @class AliVCaloTrigger
 * @brief  Virtual class to access calorimeter  (EMCAL, PHOS, PMD, FMD) trigger data
 * @author Salvatore Aiola
 */
class AliVCaloTrigger : public TNamed 
{
public:

  AliVCaloTrigger(): TNamed() {;}
  AliVCaloTrigger(const char* name, const char* title) : TNamed(name, title) {;}
  AliVCaloTrigger(const AliVCaloTrigger& ctrig);
  virtual ~AliVCaloTrigger() {;}
  AliVCaloTrigger& operator=(const AliVCaloTrigger& ctrig);
	
  virtual Bool_t       IsEmpty()                                             = 0;
  virtual void         Reset()                                               = 0;
  virtual void         Allocate(Int_t /*size*/)                              = 0;
  virtual void         DeAllocate()                                          = 0;
  
  virtual Bool_t       Add(Int_t /*col*/, Int_t /*row*/, 
                           Float_t /*amp*/, Float_t /*time*/, 
                           Int_t* /*trgtimes*/, Int_t /*ntrgtimes*/, 
                           Int_t /*trgts*/, Int_t /*trgbits*/)               = 0;
  
  virtual Bool_t       Add(Int_t /*col*/, Int_t /*row*/, 
                           Float_t /*amp*/, Float_t /*time*/, 
                           Int_t* /*trgtimes*/, Int_t /*ntrgtimes*/, 
                           Int_t /*trgts*/, Int_t /*subr*/, Int_t /*trgbit*/)= 0;

  
  virtual void         SetL1Threshold(Int_t /*i*/, Int_t /*thr*/)            = 0;
  virtual void         SetL1Threshold(Int_t /*i*/, Int_t /*j*/, Int_t /*th*/)= 0;
  
  virtual void         SetL1V0(const Int_t* /*v*/)                           = 0;
  virtual void         SetL1V0(Int_t /*i*/, const Int_t* /*v*/)              = 0;
  
  virtual void         SetL1FrameMask(Int_t /*m*/)                           = 0;
  virtual void         SetL1FrameMask(Int_t /*i*/, Int_t /*m*/)              = 0;

  /**
   * @brief Access to position of the current fastor channel
   * @param[out] col Column of the current fastor in the detector plane
   * @param[out] row Row of the current fastor in the detector plane
   */
  virtual void         GetPosition(Int_t& /*col*/, Int_t& /*row*/)    const  = 0;

  /**
   * @brief Access to L0-amplitude of the current fastor channel
   * @param[out] amp L0-amplitude for the given fastor channel
   */
  virtual void         GetAmplitude(Float_t& /*amp*/)                 const  = 0;
  virtual void         GetTime(Float_t& /*time*/)                     const  = 0;
  
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
  virtual void         GetTriggerBits(Int_t& /*bits*/)                const  = 0;

  /**
   * @brief Get the number of L0 times for the current patch
   *
   * Level0 times are handled per L0 patch. Indexing is different
   * with respect to the fastor indexing.
   *
   * @param[out] ntimes Number of level0
   */
  virtual void         GetNL0Times(Int_t& /*ntimes*/)                 const  = 0;

  /**
   * @brief Get level0 times for the current L0 patch
   * @param times L0 times for the current L0 patch
   */
  virtual void         GetL0Times(Int_t* /*times*/)                   const  = 0;

  /**
   * @brief Get the number of entries in the trigger data
   * @return Number of entries
   */
  virtual Int_t        GetEntries()                                   const  = 0;

  /**
   * @brief Get the L1 time sums (L1 ADC values) for the current fastor
   * @param[out] timesum L1 timesums for the current fastor
   */
  virtual void         GetL1TimeSum(Int_t& /*timesum*/)               const  = 0;

  /**
   * @brief Get the L1 time sums (L1 ADC values) for the current fastor
   * @return L1 timesums for the current fastor
   */
  virtual Int_t        GetL1TimeSum()                                 const  = 0;
  
  virtual void         GetL1SubRegion(  Int_t& /*subreg*/)            const  = 0;
  virtual Int_t        GetL1SubRegion()                               const  = 0;
  
  virtual Int_t        GetL1Threshold(Int_t /*i*/)                    const  = 0;
  virtual Int_t        GetL1Threshold(Int_t /*i*/, Int_t /*j*/)       const  = 0;
  
  virtual Int_t        GetL1V0(Int_t /*i*/)                           const  = 0;
  virtual Int_t        GetL1V0(Int_t /*i*/, Int_t /*j*/)              const  = 0;
  
  virtual Int_t        GetL1FrameMask()                               const  = 0;
  virtual Int_t        GetL1FrameMask(Int_t /*i*/)                    const  = 0;
 
  virtual Int_t        GetMedian(Int_t /*i*/)                         const  = 0;
  
  virtual Int_t        GetTriggerBitWord()                            const  = 0;
  virtual void         GetTriggerBitWord(Int_t& /*bw*/ )              const  = 0;

  /**
   * @brief Forward to next trigger entry (fastor / L0 patch)
   * @return True if successful (next entry existing), false otherwise (was already at the end of the buffer)
   */
  virtual Bool_t       Next()                                                = 0;
  virtual void         Copy(TObject& obj)                             const     ;

  virtual void         Print(const Option_t* /*opt*/)                 const  = 0;
  
private:

  /// \cond CLASSIMP
  ClassDef(AliVCaloTrigger, 0)
  /// \endcond
};
#endif //ALIVCALOTRIGGER_H

