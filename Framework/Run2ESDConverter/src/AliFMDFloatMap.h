#ifndef ALIFMDFLOATMAP_H
#define ALIFMDFLOATMAP_H
/* Copyright(c) 1998-2000, ALICE Experiment at CERN, All rights
 * reserved. 
 *
 * See cxx source for full Copyright notice                               
 */
#ifndef ALIFMDMAP_H
# include "AliFMDMap.h"
#endif
//____________________________________________________________________
//
// Array of floats indexed by strip identifier.
// the floats are indexed by the coordinates 
//     DETECTOR # (1-3)
//     RING ID    ('I' or 'O', any case)
//     SECTOR #   (0-39)
//     STRIP #    (0-511)
//
class AliFMDFloatMap : public AliFMDMap
{
public:
  AliFMDFloatMap();
  AliFMDFloatMap(Int_t  maxDet, 
		 Int_t  maxRing= 0, 
		 Int_t  maxSec = 0, 
		 Int_t  maxStr = 0);
  AliFMDFloatMap(const AliFMDMap& o);
  AliFMDFloatMap(const AliFMDFloatMap& o);
  virtual ~AliFMDFloatMap() { delete [] fData;fData = 0; }
  AliFMDFloatMap& operator=(const AliFMDFloatMap& o);
  virtual void Reset(const Float_t& v=Float_t());
  virtual Float_t& operator()(UShort_t det,
			      Char_t   ring,
			      UShort_t sec,
			      UShort_t str);
  virtual const Float_t& operator()(UShort_t det,
				    Char_t   ring,
				    UShort_t sec,
				    UShort_t str) const;
  Float_t* Data() const { return fData; }
  void Print(Option_t* option="%8.4f") const { AliFMDMap::Print(option); }
  void* Ptr() const { return reinterpret_cast<void*>(fData); }
protected:
  Int_t    MaxIndex() const { return fTotal; }
  Float_t  AtAsFloat(Int_t i) const { return fData[i]; } 
  Float_t& AtAsFloat(Int_t i)       { return fData[i]; } 
  Bool_t   IsFloat() const { return kTRUE; }

  Int_t   fTotal;  // Total number of entries
  Float_t* fData;   //[fTotal]
  ClassDef(AliFMDFloatMap,3) // Map of floats
};

#endif
//____________________________________________________________________
//
// Local Variables:
//   mode: C++
// End:
//

