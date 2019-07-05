// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  FlatObject.h
/// \brief Definition of FlatObject class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEOW_GPUCOMMON_TPCFASTTRANSFORMATION_FLATOBJECT_H
#define ALICEOW_GPUCOMMON_TPCFASTTRANSFORMATION_FLATOBJECT_H

#undef NDEBUG

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstddef>
#include <memory>
#include <cstring>
#include <cassert>
#endif
#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"

//#define GPUCA_GPUCODE // uncomment to test "GPU" mode

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The FlatObject class represents base class for flat objects.
/// Objects may contain variable-size data, stored in a buffer.
/// The data may contain pointers to the buffer inside.
/// The buffer can be internal, placed in mFlatBufferContainer, or external.
///
/// Important:
/// All methods of the FlatObject are marked "protected" in order to not let users to call them directly.
/// They all should be reimplemented in daughter classes,
/// because only daughter class knows how to reset all pointers in the data buffer.
/// This is an unusual decision. Normally, to avoid confusion one should just mark methods of a base class  virtual or abstract.
/// But this right solution invokes use of a virtual table and complicates porting objects to other machines.
/// Therefore:  no virtual functions but protected methods.
///
/// == Object construction.
///
/// This base class performs some basic operations, like setting initialisation flags,
/// allocating / releasing memory etc. As no virtual methods are involved,
/// the rest should be done manually in daughter classes.
///
/// It is assumed, that a daughter object may be complicated
/// and can not be constructed by just a call of its c++ constructor.
/// May be it needs to wait for something else to be initialized first. Like a database or so.
///
/// Therefore the object may find itself in some intermediate half-constructed state,
/// where its data stays in private temporary arrays and can not be yet copied to the flat buffer.
///
/// To deal with it, some extra control on the initialization flow is provided.
///
/// The object can be constructed either
/// a) by calling
///    void startConstruction();
///    ... do something ..
///    void finishConstruction( int flatBufferSize );
///
/// b) or by cloning from another constructed(!) object:
///  obj.CloneFromObject(..)
///
/// A new obect is by default in "Not Constructed" state, operations with its buffer will cause an error.
///
///
/// == Making an internal buffer external
//
/// option a)
///  std::unique_ptr<char[]> p = obj.releaseInternalBuffer();
///  ..taking care on p..
///
/// option b)
///  std::unique_ptr<char[]> p( new char[obj.GetBufferSize()] );
///  obj.moveBufferTo( p.get() );
///  ..taking care on p..
///
/// option c)
///  std::unique_ptr<char[]> p( new char[obj.GetBufferSize()] );
///  obj.cloneFromObject(obj, p.get() );
///  ..taking care on p..
///
/// === Making an external buffer internal
///
/// option a)
/// obj.cloneFromObject( obj, nullptr );
///
/// option b)
/// obj.moveBufferTo( nullptr );
///
/// == Moving an object to other machine.
///
/// This only works when the buffer is external.
/// The object and its buffer are supposed to be bit-wise ported to a new place.
/// And they need to find each other there.
/// There are 2 options:
///
/// option a) The new buffer location is only known after the transport. In this case call:
///  obj.setActualBufferAddress( new buffer address )
/// from the new place.
///
/// option b) The new buffer location is known before the transport (case of porting to GPU). In this case call:
///   obj.setFutureBufferAddress( char* futureFlatBufferPtr );
///  before the transport. The object will be ready-to-use right after the porting.
///

#ifndef GPUCA_GPUCODE // code invisible on GPU

template <typename T>
T* resizeArray(T*& ptr, int oldSize, int newSize, T* newPtr = nullptr)
{
  // Resize array pointed by ptr. T must be a POD class.
  // If the non-null newPtr is provided, use it instead of allocating a new one.
  // In this case it is up to the user to ensure that it has at least newSize slots allocated.
  // Return original array pointer, so that the user can manage previously allocate memory
  if (oldSize < 0) {
    oldSize = 0;
  }
  if (newSize > 0) {
    if (!newPtr) {
      newPtr = new T[newSize];
    }
    int mcp = std::min(newSize, oldSize);
    std::memmove(newPtr, ptr, mcp * sizeof(T));
    if (newSize > oldSize) {
      std::memset(newPtr + mcp, 0, (newSize - oldSize) * sizeof(T));
    }
  }
  T* oldPtr = ptr;
  ptr = newPtr;
  return oldPtr;
}

template <typename T>
T** resizeArray(T**& ptr, int oldSize, int newSize, T** newPtr = nullptr)
{
  // Resize array of pointers pointed by ptr.
  // If the non-null newPtr is provided, use it instead of allocating a new one.
  // In this case it is up to the user to ensure that it has at least newSize slots allocated.
  // Return original array pointer, so that the user can manage previously allocate memory
  if (oldSize < 0) {
    oldSize = 0;
  }
  if (newSize > 0) {
    if (!newPtr) {
      newPtr = new T*[newSize];
    }
    int mcp = std::min(newSize, oldSize);
    std::memmove(newPtr, ptr, mcp * sizeof(T*));
    if (newSize > oldSize) {
      std::memset(newPtr + mcp, 0, (newSize - oldSize) * sizeof(T*));
    }
  }
  T** oldPtr = ptr;
  ptr = newPtr;
  return oldPtr;
}

#endif //! GPUCA_GPUCODE

class FlatObject
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor / destructor
  FlatObject() CON_DEFAULT;
  ~FlatObject();
  FlatObject(const FlatObject&) CON_DELETE;
  FlatObject& operator=(const FlatObject&) CON_DELETE;

 protected:
  /// _____________  Memory alignment  __________________________

  /// Gives minimal alignment in bytes required for the class object
  static constexpr size_t getClassAlignmentBytes() { return 8; }

  /// Gives minimal alignment in bytes required for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() { return 8; }

  /// _____________ Construction _________

  /// Starts the construction procedure. A daughter class should reserve temporary memory.
  void startConstruction();

  /// Finishes construction: creates internal flat buffer.
  /// A daughter class should put all created variable-size members to this buffer
  ///
  void finishConstruction(int flatBufferSize);

  /// Set the object to NotConstructed state, release the buffer
  void destroy();

/// Initializes from another object, copies data to newBufferPtr
/// When newBufferPtr==nullptr, an internal container will be created, the data will be copied there.
/// A daughter class should relocate pointers inside the buffer.
///
#ifndef GPUCA_GPUCODE
  void cloneFromObject(const FlatObject& obj, char* newFlatBufferPtr);
#endif // !GPUCA_GPUCODE

  /// _____________  Methods for making the data buffer external  __________________________

  // Returns an unique pointer to the internal buffer with all the rights. Makes the internal container variable empty.
  char* releaseInternalBuffer();

/// Sets buffer pointer to the new address, move the buffer content there.
/// A daughter class must relocate all the pointers inside th buffer
#ifndef GPUCA_GPUCODE
  void moveBufferTo(char* newBufferPtr);
#endif // !GPUCA_GPUCODE

  /// _____________  Methods for moving the class with its external buffer to another location  __________________________

  /// Sets the actual location of the external flat buffer after it has been moved (i.e. to another maschine)
  /// It sets  mFlatBufferPtr to actualFlatBufferPtr.
  /// A daughter class should later update all the pointers inside the buffer in the new location.
  ///
  void setActualBufferAddress(char* actualFlatBufferPtr);

  /// Sets a future location of the external flat buffer before moving it to this location (i.e. when copying to GPU).
  ///
  /// The object can be used immidiatelly after the move, call of setActualFlatBufferAddress() is not needed.
  ///
  /// A daughter class should already relocate all the pointers inside the current buffer to the future location.
  /// It should not touch memory in the future location, since it may be not yet available.
  ///
  /// !!! Information about the actual buffer location will be lost.
  /// !!! Most of the class methods may be called only after the buffer will be moved to its new location.
  /// !!! To undo call setActualFlatBufferAddress()
  ///
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Utilities  _______________________________________________

 public:
  /// Gives size of the flat buffer
  size_t getFlatBufferSize() const { return mFlatBufferSize; }

  /// Gives pointer to the flat buffer
  const char* getFlatBufferPtr() const { return mFlatBufferPtr; }

  /// Tells if the object is constructed
  bool isConstructed() const { return (mConstructionMask & (unsigned int)ConstructionState::Constructed); }

  /// Tells if the buffer is internal
  bool isBufferInternal() const { return ((mFlatBufferPtr != nullptr) && (mFlatBufferPtr == mFlatBufferContainer)); }

  // Adopt an external pointer as internal buffer
  void adoptInternalBuffer(char* buf);

  // Hard reset of internal pointer to nullptr without deleting (needed copying an object without releasing)
  void clearInternalBufferPtr();

  /// _______________  Generic utilities  _______________________________________________

 public:
  /// Increases given size to achieve required alignment
  static size_t alignSize(size_t sizeBytes, size_t alignmentBytes)
  {
    auto res = sizeBytes % alignmentBytes;
    return res ? sizeBytes + (alignmentBytes - res) : sizeBytes;
  }

  /// Relocates a pointer inside a buffer to the new buffer address
  template <class T>
  static T* relocatePointer(const char* oldBase, char* newBase, const T* ptr)
  {
    return (ptr != nullptr) ? reinterpret_cast<T*>(newBase + (reinterpret_cast<const char*>(ptr) - oldBase)) : nullptr;
  }

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU
  void printC() const
  {
    bool lfdone = false;
    for (int i = 0; i < mFlatBufferSize; i++) {
      unsigned char v = mFlatBufferPtr[i];
      lfdone = false;
      printf("0x%02x ", v);
      if (i && ((i + 1) % 20) == 0) {
        printf("\n");
        lfdone = true;
      }
    }
    if (!lfdone) {
      printf("\n");
    }
  }
#endif //! GPUCA_GPUCODE

 protected:
  /// _______________  Data members  _______________________________________________

  /// Enumeration of construction states
  enum ConstructionState : unsigned int {
    NotConstructed = 0x0, ///< the object is not constructed
    Constructed = 0x1,    ///< the object is constructed, temporary memory is released
    InProgress = 0x2      ///< construction started: temporary  memory is reserved
  };

  int mFlatBufferSize = 0;                                            ///< size of the flat buffer
  unsigned int mConstructionMask = ConstructionState::NotConstructed; ///< mask for constructed object members, first two bytes are used by this class
  char* mFlatBufferContainer = nullptr;                               //[mFlatBufferSize]  Optional container for the flat buffer
  char* mFlatBufferPtr = nullptr;                                     //!  Pointer to the flat buffer

  ClassDefNV(FlatObject, 1);
};

/// ========================================================================================================
///
///       Inline implementations of methods
///
/// ========================================================================================================

#ifndef GPUCA_GPUCODE // code invisible on GPU
inline FlatObject::~FlatObject()
{
  destroy();
}

inline void FlatObject::startConstruction()
{
  /// Starts the construction procedure. A daughter class should reserve temporary memory.
  destroy();
  mConstructionMask = ConstructionState::InProgress;
}

inline void FlatObject::destroy()
{
  /// Set the object to NotConstructed state, release the buffer
  mFlatBufferSize = 0;
  delete[] mFlatBufferContainer;
  mFlatBufferPtr = mFlatBufferContainer = nullptr;
  mConstructionMask = ConstructionState::NotConstructed;
}

inline void FlatObject::finishConstruction(int flatBufferSize)
{
  /// Finishes construction: creates internal flat buffer.
  /// A daughter class should put all created variable-size members to this buffer

  assert(mConstructionMask & (unsigned int)ConstructionState::InProgress);

  mFlatBufferSize = flatBufferSize;
  mFlatBufferPtr = mFlatBufferContainer = new char[mFlatBufferSize];

  memset((void*)mFlatBufferPtr, 0, mFlatBufferSize); // just to avoid random behavior in case of bugs

  mConstructionMask = (unsigned int)ConstructionState::Constructed; // clear other possible construction flags
}

inline void FlatObject::cloneFromObject(const FlatObject& obj, char* newFlatBufferPtr)
{
  /// Initializes from another object, copies data to newBufferPtr
  /// When newBufferPtr==nullptr, the internal container will be created, the data will be copied there.
  /// obj can be *this (provided it does not own its buffer AND the external buffer is provided, which means
  // that we want to relocate the obj to external buffer)

  assert(obj.isConstructed());

  // providing *this with internal buffer as obj makes sens only if we want to conver it to object with PROVIDED external buffer
  assert(!(!newFlatBufferPtr && obj.mFlatBufferPtr == mFlatBufferPtr && obj.isBufferInternal()));

  char* oldPtr = resizeArray(mFlatBufferPtr, mFlatBufferSize, obj.mFlatBufferSize, newFlatBufferPtr);

  if (isBufferInternal()) {
    delete[] oldPtr; // delete old buffer if owned
  }
  mFlatBufferSize = obj.mFlatBufferSize;
  mFlatBufferContainer = newFlatBufferPtr ? nullptr : mFlatBufferPtr; // external buffer is not provided, make object to own the buffer
  std::memcpy(mFlatBufferPtr, obj.mFlatBufferPtr, obj.mFlatBufferSize);
  mConstructionMask = (unsigned int)ConstructionState::Constructed;
}
#endif

inline char* FlatObject::releaseInternalBuffer()
{
  // returns an pointer to the internal buffer. Makes the internal container variable empty.
  char* contPtr = mFlatBufferContainer;
  mFlatBufferContainer = nullptr;
  return contPtr;
}

inline void FlatObject::adoptInternalBuffer(char* buf)
{
  // buf becomes the new internal buffer, after it was already set as new setActualBufferAddress
  assert(mFlatBufferPtr == buf);
  mFlatBufferContainer = buf;
}

inline void FlatObject::clearInternalBufferPtr()
{
  // we just release the internal buffer ressetting it to nullptr
  mFlatBufferContainer = nullptr;
}

#ifndef GPUCA_GPUCODE // code invisible on GPU
inline void FlatObject::moveBufferTo(char* newFlatBufferPtr)
{
  /// sets buffer pointer to the new address, move the buffer content there.
  if (newFlatBufferPtr == mFlatBufferContainer) {
    return;
  }
  resizeArray(mFlatBufferPtr, mFlatBufferSize, mFlatBufferSize, newFlatBufferPtr);
  delete[] mFlatBufferContainer;
  mFlatBufferContainer = nullptr;
  if (!newFlatBufferPtr) { // resizeArray has created own array
    mFlatBufferContainer = mFlatBufferPtr;
  }
}
#endif

inline void FlatObject::setActualBufferAddress(char* actualFlatBufferPtr)
{
  /// Sets the actual location of the external flat buffer after it has been moved (i.e. to another maschine)
  ///
  /// It sets  mFlatBufferPtr to actualFlatBufferPtr.
  /// A daughter class should update all the pointers inside the buffer in the new location.

  assert(!isBufferInternal());
  mFlatBufferPtr = actualFlatBufferPtr;
#ifndef GPUCA_GPUCODE            // code invisible on GPU
  delete[] mFlatBufferContainer; // for a case..
#endif                           // !GPUCA_GPUCODE
  mFlatBufferContainer = nullptr;
}

inline void FlatObject::setFutureBufferAddress(char* futureFlatBufferPtr)
{
  /// Sets a future location of the external flat buffer before moving it to this location.
  ///
  /// A daughter class should already reset all the pointers inside the current buffer to the future location
  /// without touching memory in the future location.

  assert(!isBufferInternal());
  mFlatBufferPtr = futureFlatBufferPtr;
#ifndef GPUCA_GPUCODE            // code invisible on GPU
  delete[] mFlatBufferContainer; // for a case..
#endif                           // !GPUCA_GPUCODE
  mFlatBufferContainer = nullptr;
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
