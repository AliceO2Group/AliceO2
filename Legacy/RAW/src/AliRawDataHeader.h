#ifndef ALIRAWDATAHEADER_H
#define ALIRAWDATAHEADER_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

struct AliRawDataHeader {
  AliRawDataHeader() :
    fSize(0xFFFFFFFF), 
    fWord2(2<<24),
    fEventID2(0),
    fAttributesSubDetectors(0),
    fStatusMiniEventID(0x10000),  // status bit 4: no L1/L2 trigger information
    fTriggerClassLow(0),
    fROILowTriggerClassHigh(0),
    fROIHigh(0)
  {}

  // Adding virtual destructor breaks
  // C++ struct backward compatibility
  // Do not uncomment the line below!!!
  //  virtual ~AliRawDataHeader() {}

  UShort_t  GetEventID1() const
    {
      return (UShort_t)( fWord2 & 0xFFF );
    };

  UInt_t  GetEventID2() const
    {
      return fEventID2;
    };

  UChar_t   GetL1TriggerMessage() const
    {
      return (UChar_t)( (fWord2 >> 14) & 0xFF );
    };

  UChar_t   GetVersion() const
    {
      return (UChar_t)( (fWord2 >> 24) & 0xFF );
    };

  UChar_t   GetAttributes() const 
    {return (fAttributesSubDetectors >> 24) & 0xFF;};
  Bool_t    TestAttribute(Int_t index) const
    {return (fAttributesSubDetectors >> (24 + index)) & 1;};
  void      SetAttribute(Int_t index)
    {fAttributesSubDetectors |= (1 << (24 + index));};
  void      ResetAttribute(Int_t index)
    {fAttributesSubDetectors &= (0xFFFFFFFF ^ (1 << (24 + index)));};
  UInt_t    GetSubDetectors() const
    {return fAttributesSubDetectors & 0xFFFFFF;};

  UInt_t    GetStatus() const
    {return (fStatusMiniEventID >> 12) & 0xFFFF;};
  UInt_t    GetMiniEventID() const
    {return fStatusMiniEventID & 0xFFF;};

  ULong64_t GetTriggerClasses() const
  {return (((ULong64_t) (fROILowTriggerClassHigh & 0x3FFFF)) << 32) | fTriggerClassLow;}
  ULong64_t GetTriggerClassesNext50() const
  {return 0;}
  ULong64_t GetROI() const
  {return (((ULong64_t) fROIHigh) << 4) | ((fROILowTriggerClassHigh >> 28) & 0xF);}

  void      SetTriggerClass(ULong64_t mask)
    {fTriggerClassLow = (UInt_t)(mask & 0xFFFFFFFF);  // low bits of trigger class
     fROILowTriggerClassHigh = (UInt_t)((mask >> 32) & 0x3FFFF); // low bits of ROI data (bits 28-31) and high bits of trigger class (bits 0-17)
    };

  UInt_t    fSize;              // size of the raw data in bytes
  UInt_t    fWord2;             // bunch crossing, L1 trigger message and fomrat version
  UInt_t    fEventID2;          // orbit number
  UInt_t    fAttributesSubDetectors; // block attributes (bits 24-31) and participating sub detectors
  UInt_t    fStatusMiniEventID; // status & error bits (bits 12-27) and mini event ID (bits 0-11)
  UInt_t    fTriggerClassLow;   // low bits of trigger class
  UInt_t    fROILowTriggerClassHigh; // low bits of ROI data (bits 28-31) and high bits of trigger class (bits 0-17)
  UInt_t    fROIHigh;           // high bits of ROI data
};

#endif
