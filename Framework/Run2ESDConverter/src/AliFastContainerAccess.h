#ifndef ALIFASTCONTAINERACCESS_H
#define ALIFASTCONTAINERACCESS_H

#include <TMatrixT.h>
#include <cassert>

namespace AliFastContainerAccess {

// a fast access to ROOT's TMatrix class
// without boundary checks nor R_ASSERT as enforced by ROOT
// boundary checks can be enabled by compiling without -DNDEBUG
template <typename T>
inline
T TMatrixFastAt(TMatrixT<T> const &m, int rown, int coln){
 const Int_t arown = rown - m.GetRowLwb();
 const Int_t acoln = coln - m.GetColLwb();
#ifndef NDEBUG
// put boundary checks here
#endif
 T const *entries = m.TMatrixT<T>::GetMatrixArray();

 // verify correctness w.r.t to original implementation
 assert(entries[arown*m.GetNcols() + acoln] == m(rown,coln));

 return entries[arown*m.GetNcols() + acoln];
}

// fast access to a reference to elements of ROOT's TMatrix class
template <typename T>
inline
T &TMatrixFastAtRef(TMatrixT<T> &m, int rown, int coln){
 const Int_t arown = rown - m.GetRowLwb();
 const Int_t acoln = coln - m.GetColLwb();
#ifndef NDEBUG
// put boundary checks here
#endif
 T *entries = m.TMatrixT<T>::GetMatrixArray();

 // verify correctness w.r.t to original implementation
 assert(entries[arown*m.GetNcols() + acoln] == m(rown,coln));

 return entries[arown*m.GetNcols() + acoln];
}



}


#endif // ALIFASTCONTAINERACCESS_H
