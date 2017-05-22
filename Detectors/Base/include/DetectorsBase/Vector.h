#ifndef O2_BASE_VECTOR_H
#define O2_BASE_VECTOR_H

#include <TTree.h>
#include <TBranch.h>
#include <TBranchElement.h>
#include <TBuffer.h>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>
#include <FairLogger.h>


namespace o2 {
namespace Base {

 template <class T, class H>
   class Vector {//:public TObject {
   
 public:
   
   using sizeType = std::int64_t;
   
  struct Header {
    H         userInfo;         // user assigned data info
    int       expandPolicy;     // user assigned policy: n>0 -> new=old+n, n<=0 -> new=2*(old+n)
    sizeType  sizeInBytes;      // total booked size
    sizeType  nObjects;         // number of objects stored
  };
  // main constructor
  Vector(sizeType iniSize=0, int expPol=-100);
  
  // construct from received raw pointer on the existing buffer (create a copy of the buffer, ptr ownership unchanged)
  Vector(const char* ptr, sizeType nbytes=-1) : mPtr(nullptr) {recreate(ptr,nbytes);}

  // construct from received pointer on the existing buffer (assuming ownership over the buffer)
  Vector(std::unique_ptr<char[]> ptr, sizeType nbytes=-1) : mPtr(nullptr) {adopt(std::move(ptr),nbytes);}
  
  // recreate buffer copy from received pointer,  ptr ownership unchanged
  void  recreate(const char* ptr, sizeType nbytes=-1);

  // adopt buffer from received pointer by assuming its ownership
  void  adopt(std::unique_ptr<char[]> ptr, sizeType nbytes=-1);
  
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
  
  /// clear content w/o changing capacity
  void  clear(/*bool calldestructor=true*/);

  /// book space for objects and aux data
  void  reserve(sizeType n=1000);

  /// expand space for new objects
  void  expand();

  /// get raw pointer on the buffer, can be sent to another process
  char*               getPtr()    const {return mPtr.get();}

  /// returns the buffer pointer (releases the ownership) and reset the container
  std::unique_ptr<char[]> release();

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
  // for some reason using static constexpr leads to problem in linking the testVector
  //  static constexpr sizeType dataOffset() = sizeof(Header) + sizeof(Header)%alignof(T);

  
  std::unique_ptr<char[]> mPtr;                //! pointer on continuos block containing full object data
  std::vector<T> mVecForTree;                  //! vector for writing objects into the tree
  
  ClassDef(Vector,1)
};



//-------------------------------------------------------------------
template <class T, class H>
  Vector<T,H>::Vector(sizeType iniSize, int expPol) : mPtr(nullptr), mVecForTree()
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
  void Vector<T,H>::recreate(const char* ptr, sizeType nbytes)
{
  /**
   * recreates container buffer from provided raw pointer 
   * ptr: pointer extracted from the container of simular type and owned by it, using e.g. getPtr() method.
   * nbytes: expected size of the buffer in bytes, used for error check
   */

  if (ptr==nullptr) {
    std::ostringstream strErr;
    strErr << "invalid arguments: ptr is null" << std::endl;
    throw strErr.str();
  }
  
  if (nbytes>=0 && (nbytes<dataOffset())) {
    std::ostringstream strErr;
    strErr << "invalid arguments: " << "wrong size " << nbytes <<
      " at least " << dataOffset() << " expected" << std::endl;
    throw  strErr.str();
  }
  
  const sizeType nbDecoded = (reinterpret_cast<const Header*>(ptr))->sizeInBytes;
  if (nbytes>=0 && (nbytes!=nbDecoded)) {
    std::ostringstream strErr;
    strErr << "invalid arguments: " <<" supplied size " << nbytes <<  " differs from decoded size " << nbDecoded << std::endl;
    throw strErr.str();
  }

  // create a copy of the buffer
  mPtr = std::make_unique<char[]>(nbDecoded);
  std::copy(ptr,ptr+nbDecoded,mPtr.get());
  
}

//-------------------------------------------------------------------
template <class T, class H>
  void Vector<T,H>::adopt(std::unique_ptr<char[]> ptr, sizeType nbytes)
{
  /**
   * recreates container buffer from provided pointer assuming its ownership
   * ptr:    pointer extracted from the container of simular type using e.g. release() methods
   * nbytes: expected size of the buffer in bytes, used for error check
   */
  
  if (ptr==nullptr) {
    std::ostringstream strErr;
    strErr << "invalid arguments: ptr is null" << std::endl;
    throw strErr.str();
  }

  if (nbytes>=0 && (nbytes<dataOffset())) {
    std::ostringstream strErr;
    strErr << "invalid arguments: " << "wrong size " << nbytes <<
      " at least " << dataOffset() << " expected" << std::endl;
    throw  strErr.str();
  }

  sizeType nbDecoded = (reinterpret_cast<Header*>(ptr.get()))->sizeInBytes;
  if (nbytes>=0 && (nbytes!=nbDecoded)) {
    std::ostringstream strErr;
    strErr << "invalid arguments: " <<" supplied size " << nbytes <<  " differs from decoded size " << nbDecoded << std::endl;
    throw strErr.str();
  }
  
  mPtr = std::move(ptr); // transfer ownership
  
}

//-------------------------------------------------------------------
template <class T, class H>
  void Vector<T,H>::reserve(sizeType n)
{
  /**
   * Resize container to size n. 
   * Existing content is preserved, but truncated to n if n<current_n_objects
   */
  if (n<0) n = 0;
  sizeType bookSize = dataOffset() + n*sizeof(T);
  //  auto tmpPtr = std::move(mPtr);
  std::unique_ptr<char[]> tmpPtr = std::make_unique<char[]>(bookSize);
  if (mPtr!=nullptr) std::copy(mPtr.get(), mPtr.get()+std::min(bookSize,sizeInBytes()), tmpPtr.get());  
  mPtr.swap(tmpPtr);
  if (n<getHeader()->nObjects) getHeader()->nObjects = n;
  getHeader()->sizeInBytes = bookSize;
}

//-------------------------------------------------------------------
template <class T, class H>
  void Vector<T,H>::setUserInfo(const H& v)
{
  /**
   * Sets data identification field
   */
  getHeader()->userInfo = v;
  //  std::copy_n(&v,1, &getHeader()->userInfo);
}

//-------------------------------------------------------------------
template <class T, class H>
  void Vector<T,H>::expand()
{
  /**
   * expand container according to expansion policy
   */
  auto oldSize = capacity();
  auto newSize = getExpandPolicy()<0 ?
    2*std::max(oldSize-getExpandPolicy(),static_cast<sizeType>(1)) :
    oldSize+getExpandPolicy();
  reserve(newSize);
}

//-------------------------------------------------------------------
template <class T, class H>
  void Vector<T,H>::clear(/*bool calldestructor*/)
{
  /**
   * clear content w/o changing capacity, if requested, explicitly delete objects
   */
  /*
  if (calldestructor) {
    T *objB = back(), *objF = front();
    while (objB>=objF) {
      objB->~T();
      objB--;
    }
  }
  */
  getHeader()->nObjects = 0;
}

//-------------------------------------------------------------------
template <class T, class H>
  inline T* Vector<T,H>::nextFreeSlot()
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
  T* Vector<T,H>::push_back(const T& obj)
{
  /**
   * Create copy of the object in the end of the container
   * Return pointer on new object
   */
  T* slot = nextFreeSlot();
  //new(slot) T(obj);
  std::copy_n(&obj,1, slot);
  getHeader()->nObjects++;
  return slot;
}

//-------------------------------------------------------------------
template<class T, class H> template <typename ...Args>
  T* Vector<T,H>::emplace_back(Args&&... args) 
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
  std::unique_ptr<char[]> Vector<T,H>::release()
{
  /**
   * returns a buffer pointer (releasing the ownership) and reset the container
   * 
   */
  auto ptr = std::move(mPtr);
  reserve(0);
  setUserInfo( reinterpret_cast<Header*>(ptr.get())->userInfo );
  setExpandPolicy( reinterpret_cast<Header*>(ptr.get())->expandPolicy );
  return ptr;
}


//-------------------------------------------------------------------
template<class T, class H>
  void Vector<T,H>::Streamer(TBuffer &b)
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
  void Vector<T,H>::AddToTree(TTree* tree, const std::string& brName)
{
  /**
   * Add T-class objects of the container to std::vector<T> branch (create if needed) in the tree.
   * Note that the Tree::Fill has to be called by user afterwards
   *
   */
  
  if (tree==nullptr || !brName.size() ) {
    std::ostringstream strErr;
    strErr << "invalid arguments:" << " tree: " << tree << " branchName: " << brName << std::endl;
    throw strErr.str();
  }

  TBranch* br = tree->GetBranch(brName.data());
  if (!br) { // need to create branch
    br = tree->Branch(brName.data(), &mVecForTree);
    //    std::cout << "Added branch "<< brName << " to tree " << tree->GetName() << std::endl;
  }
  
  // fill objects
  mVecForTree.clear();
  auto n = size();
  for (auto i=0;i<n;i++) {
    mVecForTree.push_back( *(*this)[i] );
  } 
}

}
}

#endif
