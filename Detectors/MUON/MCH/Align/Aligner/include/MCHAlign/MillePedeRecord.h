#ifndef ALICEO2_MCH_MILLEPEDERECORD_H
#define ALICEO2_MCH_MILLEPEDERECORD_H

#include <TObject.h>

namespace o2
{
namespace mch
{

class MillePedeRecord : public TObject
{
 public:
  MillePedeRecord();
  MillePedeRecord(const MillePedeRecord& src);
  MillePedeRecord& operator=(const MillePedeRecord& rhs);

  ~MillePedeRecord() override;
  void Reset();
  void Print(const Option_t* opt = "") const override;

  int GetSize() const { return fSize; }
  int* GetIndex() const { return fIndex; }
  int GetIndex(int i) const { return fIndex[i]; }

  void GetIndexValue(int i, int& ind, double& val) const
  {
    ind = fIndex[i];
    val = fValue[i];
  }

  void AddIndexValue(int ind, double val);
  void AddResidual(double val) { AddIndexValue(-1, val); }
  void AddWeight(double val) { AddIndexValue(-2, val); }
  void SetWeight(double w = 1) { fWeight = w; }
  bool IsResidual(int i) const { return fIndex[i] == -1; }
  bool IsWeight(int i) const { return fIndex[i] == -2; }
  double* GetValue() const { return fValue; }
  double GetValue(int i) const { return fValue[i]; }
  double GetWeight() const { return fWeight; }

  void MarkGroup(int id);
  int GetNGroups() const { return fNGroups; }
  int GetGroupID(int i) const { return fGroupID[i] - 1; }
  bool IsGroupPresent(int id) const;
  unsigned int GetRunID() const { return fRunID; }
  void SetRunID(unsigned int run) { fRunID = run; }

  // Aux methods
  double GetGlobalDeriv(int pnt, int indx) const;
  double GetLocalDeriv(int pnt, int indx) const;
  double GetResidual(int pnt) const;
  double GetGloResWProd(int indx) const;
  double GetWeight(int indx) const;

 protected:
  int GetDtBufferSize() const { return GetUniqueID() & 0x0000ffff; }
  int GetGrBufferSize() const { return GetUniqueID() >> 16; }
  void SetDtBufferSize(int sz) { SetUniqueID((GetGrBufferSize() << 16) + sz); }
  void SetGrBufferSize(int sz) { SetUniqueID(GetDtBufferSize() + (sz << 16)); }
  void ExpandDtBuffer(int bfsize);
  void ExpandGrBuffer(int bfsize);
  //
 protected:
  int fSize;                    // size of the record
  int fNGroups;                 // number of groups (e.g. detectors) contributing
  unsigned int fRunID;          // run ID
  unsigned short int* fGroupID; //[fNGroups] groups id's+1 (in increasing order)
  int* fIndex;                  //[fSize] index of variables
  Double32_t* fValue;           //[fSize] array of values: derivs,residuals
  Double32_t fWeight;           // global weight for the record
  //
  ClassDefOverride(MillePedeRecord, 3) // Record of track residuals and local/global deriavtives
};

//_____________________________________________________________________________________________
inline void MillePedeRecord::AddIndexValue(int ind, double val)
{
  // add new pair of index/value
  if (fSize >= GetDtBufferSize()) {
    ExpandDtBuffer(2 * (fSize + 1));
  }
  fIndex[fSize] = ind;
  fValue[fSize++] = val;
}

//_____________________________________________________________________________________________
inline bool MillePedeRecord::IsGroupPresent(int id) const
{
  // check if group is defined
  id++;
  for (int i = fNGroups; i--;) {
    if (fGroupID[i] == id) {
      return true;
    }
  }
  return false;
}

} // namespace mch
} // namespace o2

#endif
