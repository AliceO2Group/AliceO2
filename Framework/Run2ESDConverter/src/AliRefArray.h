#ifndef ALIREFARRAY_H
#define ALIREFARRAY_H
//_________________________________________________________________________
// 
//        Class for association of multiple UInt_t labels to array of UInt_t
//
// The use-case: reference from the clusterID to (multiple) trackID's using this cluster
// 
// ATTENTION: the references are provided as UInt_t, but maximum value should not exceed MAX_INT-1 (no check is done)
//
// Author: ruben.shahoyan@cern.ch
//_________________________________________________________________________

#include "TObject.h"

class AliRefArray : public TObject {
 public:
  AliRefArray();
  AliRefArray(UInt_t nelem, UInt_t depth=0);
  AliRefArray(const AliRefArray& src);
  AliRefArray& operator=(const AliRefArray& src);
  virtual ~AliRefArray();
  //
  UInt_t GetNElems()                                             const {return fNElems;}
  void   Expand(UInt_t size=0);
  Bool_t IsReferred(UInt_t from, UInt_t to)                      const;
  Bool_t HasReference(UInt_t from)                               const {return (from>=fNElems||!fElems[from]) ? kFALSE:kTRUE;}
  void   AddReference(UInt_t from, UInt_t to);
  void   AddReferences(UInt_t from, UInt_t* refs, UInt_t nref);
  UInt_t GetReferences(UInt_t from, UInt_t* refs, UInt_t maxRef) const;
  Int_t  GetReference(UInt_t from, UInt_t which)                 const;
  void   Reset();
  void   Print(Option_t* opt="")                                 const;
  void   Compactify();
  //
 protected:
  void   ExpandReferences(Int_t addSize);
  //
 protected:
  UInt_t        fNElems;               //   number of referrer elements
  UInt_t        fRefSize;              //   current size of all references
  Int_t*        fElems;                //[fNElems] array of referrers
  UInt_t*       fRefInd;               //[fRefSize] indices of next referred node
  UInt_t*       fRefBuff;              //[fRefSize] buffer of entries for referred nodes
  ClassDef(AliRefArray,1)
  //
};

//____________________________________________________________________
inline Bool_t AliRefArray::IsReferred(UInt_t from, UInt_t to)  const
{
  // check if the cluster "to" is mentioned in the references of the cluster "from"
  Int_t ref;
  if (from>=fNElems || !(ref=fElems[from])) return kFALSE;
  if (ref<0) {return (ref+int(to))==-1;}   // negative means just 1 reference: -(ref+1) is stored
  to++;
  do { if (fRefBuff[ref]==to) return kTRUE; } while((ref=fRefInd[ref])); // search intil no references left
  return kFALSE;
}

//____________________________________________________________________
inline UInt_t AliRefArray::GetReferences(UInt_t from, UInt_t* refs, UInt_t maxRef) const
{
  // extract max maxRef references for node "from" to user provided array refs
  Int_t ref;
  UInt_t nrefs=0;
  if (from>=fNElems || !(ref=fElems[from])) return 0; // no references
  if (ref<0) {refs[0] = -(1+ref); return 1;}  // just 1 reference
  do { refs[nrefs++]=fRefBuff[ref]-1; } while((ref=(int)fRefInd[ref]) && nrefs<maxRef); // search intil no references left
  return nrefs;
}

//____________________________________________________________________
inline Int_t AliRefArray::GetReference(UInt_t from, UInt_t which) const
{
  // returns reference number which (if n/a: -1)
  Int_t ref;
  if (from>=fNElems || !(ref=fElems[from])) return -1; // no references
  if (ref<0) return which ? -1 : -(1+ref);             // just 1 reference
  int ref1 = ref;
  while(which && (ref1=(int)fRefInd[ref])) {ref=ref1;which--;} // search intil no references left
  return which ? -1 : (Int_t) fRefBuff[ref]-1;
}

//____________________________________________________________________
inline void AliRefArray::AddReference(UInt_t from, UInt_t to)
{
  // add node "to" to the references of "from"
  if (from>=fNElems) Expand(from+1);
  int &ref0 = fElems[from];
  if (!ref0) {ref0 = -(++to); return;}         // 1st reference, save in situ
  //
  int chk = ref0>0 ? 1:2; // if <0 (just 1 ref.before) need to transfer both to index array
  if (!fRefInd || int(fRefInd[0])>(int(fRefSize)-chk)) ExpandReferences( fRefSize );
  UInt_t &freeSlot = fRefInd[0];
  Int_t ref = fElems[from];
  if (ref<0) { fRefInd[freeSlot]=0; fRefBuff[freeSlot] = -ref; ref = fElems[from] = freeSlot++; }
  //
  while(fRefInd[ref]) ref=fRefInd[ref]; // find last index of last entry for cluster from
  fRefBuff[freeSlot] = ++to;
  fRefInd[ref] = freeSlot++;            // register it in the indices
}


//____________________________________________________________________
inline void AliRefArray::Compactify()
{
  // prepare for storing with minimal space usage
  if (fRefInd && fRefSize>fRefInd[0]) fRefSize = fRefInd[0];
}
  
#endif
