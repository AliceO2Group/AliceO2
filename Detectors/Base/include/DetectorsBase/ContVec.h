#ifndef O2_BASE_CONTVEC_H
#define O2_BASE_CONTVEC_H

#include <TTree.h>
#include <TBranch.h>
#include <TBranchElement.h>
#include <TBuffer.h>
#include <TObject.h>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <iostream>
#include <FairLogger.h>


namespace o2 {
namespace Base {

using sizeType = int;

 template <class T, class H>
  class ContVec: public TObject {

 public:
  
  struct Header {
    H         userInfo;         // user assigned data info
    int       expandPolicy;     // user assigned policy: n>0 -> new=old+n, n<=0 -> new=2*(old+n)
    sizeType  sizeInBytes;      // total booked size
    sizeType  nObjects;         // number of objects stored
  };
  // main constructor
  ContVec(sizeType iniSize=0, int expPol=-100);

  // construct from received raw pointer on the existing buffer, see recreate comments
  ContVec(char* rawptr, bool copy, sizeType nbytes=-1)  : mPtr(nullptr) {recreate(rawptr,copy,nbytes);}

  // recreate container from received raw pointer on the existing buffer
  void  recreate(char* rawptr, bool copy, sizeType nbytes=-1);
  
  /// set/get data info (user defined)
  void  setUserInfo(const H& val);
  H&    getUserInfo()            const {return getHeader()->userInfo;}

  /// set/get expand policy, see constructor documentation
  void  setExpandPolicy(int val)       {getHeader()->expandPolicy = val;}
  int   getExpandPolicy()        const {return getHeader()->expandPolicy;}

  /// get number of objects stored
  sizeType size()                const {return getHeader()->nObjects;}

  /// get currently booked capacity
  sizeType capacity()            const {return (sizeInBytes()-dataOffset())/sizeof(T);}

  /// get i-th object pointer w/o boundary check
  T*    operator[](sizeType i)   const {return getData()+i;}

  /// get i-th object pointer with boundary check
  T*    at(sizeType i)           const {return i<size() ? (*this)[i] : nullptr;}

  /// get last object pointer
  T*    back()                   const {return size() ? (*this)[size()-1] : nullptr;}

  /// get 1st object pointer
  T*    front()                  const {return size() ? (*this)[0] : nullptr;}
  
  /// add copy of existing object to the end, return created copy pointer
  T*    push_back(const T& obj);

  /// create an object with supplied arguments in the end of the container
  template<typename ...Args> T* emplace_back(Args&&... args);
  
  /// clear content w/o changing capacity, if requested, explicitly delete objects
  void  clear(bool calldestructor=true);

  /// book space for objects and aux data
  void  reserve(sizeType n=1000);

  /// expand space for new objects
  void  expand();

  /// get raw pointer on the buffer, can be sent to another process
  char*               getPtr()    const {return mPtr.get();}

  /// returns a buffer pointer (releases the ownership) and reset the container
  char*               release();

  /// return total size in bytes
  sizeType            sizeInBytes() const {return getHeader()->sizeInBytes;}

  /// add objects stored in the container to vector<T> branch in the supplied tree
  void AddToTree(TTree* tree, const std::string& brName);
  
 protected:

  /// cast the head of container buffer to internal layout
  Header*             getHeader() const {return reinterpret_cast<Header*>(getPtr());}
  T*                  getData()   const {return reinterpret_cast<T*>(getPtr()+dataOffset());}

  /// return next free slot, for "new with placement" creation, not LValue
  T*                  nextFreeSlot();
      
 protected:

  // offset of data wrt buffer start: account for Header + padding after Header block to respect T alignment
  constexpr sizeType dataOffset() const {return sizeof(Header) + sizeof(Header)%alignof(T);};
  // for some reason using static constexpr leads to problem in linking the testContVec
  //  static constexpr sizeType dataOffset() = sizeof(Header) + sizeof(Header)%alignof(T);

  
  std::unique_ptr<char[]> mPtr;                //! pointer on continuos block containing full object data
  std::unique_ptr<std::vector<T>> mVecForTree; //! vector for writing objects into the tree
  
  ClassDef(ContVec,1)
};



//-------------------------------------------------------------------
template <class T, class H>
  ContVec<T,H>::ContVec(sizeType iniSize, int expPol) : mPtr(nullptr), mVecForTree(nullptr)
  {
    /**
     * Creates container for objects of 
     * T POD class
     * with single description field of H POD class
     * size: initial capacity of the container
     * expPol: expansion policy. new_size = expPol>0 ? oldSize+expPol : 2*max(1,oldSize+|expPol|)
     */
    static_assert(std::is_trivially_copyable<T>::value,"Object class is not trivially-copiable");
    static_assert(std::is_trivially_copyable<H>::value,"Header class is not trivially-copiable");
    reserve(iniSize);
    if (!expPol) expPol = -1;
    getHeader()->expandPolicy = expPol;
  }

//-------------------------------------------------------------------
template <class T, class H>
  void ContVec<T,H>::recreate(char* rawPtr, bool copy, sizeType nbytes)
{
  /**
   * recreates container buffer from provided raw pointer 
   * rawPtr: pointer extracted from the container of simular type using e.g. getPtr() or release() methods
   * nbytes: expected size of the buffer in bytes, used for error check
   * copy  : if true, create copy of the array (this is must if the new rawPtr is managed by other object, 
   *         e.g. was obtained via ContVec::getPtr())
   *         if false, then the new object will assume ownership over rawPtr
   */
  
  {
    std::unique_ptr<char[]> dum(mPtr.release()); // destroy old buffer if any (== delete[] mPtr.release())
  }
  
  try {
    if (rawPtr==nullptr) {
      throw "invalid arguments:";
    }
  } catch(const char* msg) {
    LOG(FATAL) << msg << " rawPtr is null" << FairLogger::endl;
  }
  try {
    if (nbytes>=0 && (nbytes<dataOffset())) {
      throw "invalid arguments:";
    }
  } catch(const char* msg) {
    LOG(FATAL) << msg <<  "wrong size " << nbytes <<  " at least " << dataOffset() << " expected" << FairLogger::endl;
  }
  
  sizeType nbDecoded = (reinterpret_cast<Header*>(rawPtr))->sizeInBytes;
  try {
    if (nbytes>=0 && (nbytes!=nbDecoded)) {
      throw "invalid arguments:";
    }
  } catch(const char* msg) {
    LOG(FATAL) << msg <<" supplied size " << nbytes <<  " differs from decoded size " << nbDecoded << FairLogger::endl;
    exit(1);      
  }
  
  if (copy) {
    mPtr.reset(new char[nbDecoded]());
    std::copy(rawPtr,rawPtr+nbDecoded,mPtr.get());
  }
  else {
    mPtr.reset(rawPtr);
  }
  
}

//-------------------------------------------------------------------
template <class T, class H>
  void ContVec<T,H>::reserve(sizeType n)
{
  /**
   * Resize container to size n. 
   * Existing content is preserved, but truncated to n if n<current_n_objects
   */
  if (n<0) n = 0;
  sizeType bookSize = dataOffset() + n*sizeof(T);
  //  auto tmpPtr = std::move(mPtr);
  std::unique_ptr<char[]> tmpPtr(new char[bookSize]());
  if (mPtr!=nullptr) std::copy(mPtr.get(), mPtr.get()+std::min(bookSize,sizeInBytes()), tmpPtr.get());  
  mPtr.swap(tmpPtr);
  if (n<getHeader()->nObjects) getHeader()->nObjects = n;
  getHeader()->sizeInBytes = bookSize;
}

//-------------------------------------------------------------------
template <class T, class H>
  void ContVec<T,H>::setUserInfo(const H& v)
{
  /**
   * Sets data identification field
   */
  getHeader()->userInfo = v;
  //  std::copy_n(&v,1, &getHeader()->userInfo);
}

//-------------------------------------------------------------------
template <class T, class H>
  void ContVec<T,H>::expand()
{
  /**
   * expand container according to expansion policy
   */
  auto oldSize = capacity();
  auto newSize = getExpandPolicy()<0 ? 2*std::max(oldSize-getExpandPolicy(),1) : oldSize+getExpandPolicy();
  reserve(newSize);
}

//-------------------------------------------------------------------
template <class T, class H>
  void ContVec<T,H>::clear(bool calldestructor)
{
  /**
   * clear content w/o changing capacity, if requested, explicitly delete objects
   */
  if (calldestructor) {
    T *objB = back(), *objF = front();
    while (objB>=objF) {
      objB->~T();
      objB--;
    }
  }
  getHeader()->nObjects = 0;
}

//-------------------------------------------------------------------
template <class T, class H>
  inline T* ContVec<T,H>::nextFreeSlot()
{
  /**
   * Return pointer on next free slot where new object will be placed.
   * If needed, autoexpands 
   */  
  if (size()==capacity()) expand();
  return (*this)[size()];
}

//-------------------------------------------------------------------
template <class T, class H>
  T* ContVec<T,H>::push_back(const T& obj)
{
  /**
   * Create copy of the object in the end of the container
   * Return pointer on new object
   */
  T* slot = nextFreeSlot();
  new(slot) T(obj);
  //std::copy_n(&obj,1, slot);
  getHeader()->nObjects++;
  return slot;
}

//-------------------------------------------------------------------
template<class T, class H> template <typename ...Args>
  T* ContVec<T,H>::emplace_back(Args&&... args) 
{
  /**
   * Create copy of the object in the end of the container
   * Return pointer on new object
   */
  T* slot = nextFreeSlot();
  new(slot) T(std::forward<Args>(args)...);
  getHeader()->nObjects++;
  return slot;
}

//-------------------------------------------------------------------
template<class T, class H>
  char* ContVec<T,H>::release()
{
  /**
   * returns a buffer pointer (releasing the ownership) and reset the container
   * 
   */
  char* ptr = mPtr.release();
  reserve(0);
  setUserInfo( reinterpret_cast<Header*>(ptr)->userInfo );
  setExpandPolicy( reinterpret_cast<Header*>(ptr)->expandPolicy );
  return ptr;
}


//-------------------------------------------------------------------
template<class T, class H>
  void ContVec<T,H>::Streamer(TBuffer &b)
{
   if (b.IsReading()) {
     Int_t n;
     b >> n;
     std::unique_ptr<char[]> dum(mPtr.release());
     mPtr.reset(new char[n]());
     b.ReadFastArray(mPtr.get(),n);
   } else {
     int bs = sizeInBytes();
     b << bs;
     b.WriteFastArray((char*)mPtr.get(), bs);
   }
}

//-------------------------------------------------------------------
template<class T, class H>
  void ContVec<T,H>::AddToTree(TTree* tree, const std::string& brName)
{
  /**
   * Add T-class objects of the container to std::vector<T> branch (create if needed) in the tree.
   * Note that the Tree::Fill has to be called by user afterwards
   *
   */
  
  try {
    if (tree==nullptr || !brName.size() ) {
      throw "invalid arguments:";
    }
  } catch(const char* msg) {
    LOG(FATAL) << msg << " tree: " << tree << " branchName: " << brName << FairLogger::endl;
  }

  if (!mVecForTree) {
    mVecForTree.reset( new  std::vector<T> );
  }
  std::vector<T>* vp = mVecForTree.get();

  TBranch* br = tree->GetBranch(brName.data());
  if (!br) { // need to create branch
    br = tree->Branch(brName.data(), vp);
    //    std::cout << "Added branch "<< brName << " to tree " << tree->GetName() << std::endl;
  }
  
  // fill objects
  vp->clear();
  auto n = size();
  for (auto i=0;i<n;i++) {
    vp->push_back( *(*this)[i] );
  } 
}

}
}

#endif
