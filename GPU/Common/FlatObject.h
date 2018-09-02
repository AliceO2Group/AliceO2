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


#ifndef ALICE_ALITPCOMMON_TPCFASTTRANSFORMATION_FLATOBJECT_H
#define ALICE_ALITPCOMMON_TPCFASTTRANSFORMATION_FLATOBJECT_H

#undef NDEBUG 

#include <stddef.h>
#include <memory>
#include <cstring>
#include <cassert>
#include "AliTPCCommonDef.h"

namespace ali_tpc_common {
namespace tpc_fast_transformation {

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
class FlatObject
{
 protected:    

  /// _____________  Constructors / destructors __________________________


  /// Default constructor: creates an empty uninitialized object
  FlatObject();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject instead
  FlatObject(const FlatObject& ) CON_DELETE;
 
  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject instead
  FlatObject &operator=(const FlatObject &)  CON_DELETE;

  /// Destructor
  ~FlatObject() CON_DEFAULT;

  
  /// _____________  Memory alignment  __________________________

  /// Gives minimal alignment in bytes required for the class object
  static constexpr size_t getClassAlignmentBytes() {return 8;}
 
  /// Gives minimal alignment in bytes required for the flat buffer
  static constexpr size_t getBufferAlignmentBytes() {return 8;}


  /// _____________ Construction _________


  /// Starts the construction procedure. A daughter class should reserve temporary memory.
  void startConstruction();
  
  /// Finishes construction: creates internal flat buffer. 
  /// A daughter class should put all created variable-size members to this buffer  
  ///
  void finishConstruction( int flatBufferSize );

  /// Set the object to NotConstructed state, release the buffer
  void destroy();

  /// Initializes from another object, copies data to newBufferPtr
  /// When newBufferPtr==nullptr, an internal container will be created, the data will be copied there. 
  /// A daughter class should relocate pointers inside the buffer.
  ///
  void cloneFromObject( const FlatObject &obj, char *newFlatBufferPtr );
  

  /// _____________  Methods for making the data buffer external  __________________________

  // Returns an unique pointer to the internal buffer with all the rights. Makes the internal container variable empty.
  std::unique_ptr<char[]> releaseInternalBuffer();

  /// Sets buffer pointer to the new address, move the buffer content there.
  /// A daughter class must relocate all the pointers inside th buffer
  void moveBufferTo( char *newBufferPtr );


  /// _____________  Methods for moving the class with its external buffer to another location  __________________________

  
  /// Sets the actual location of the external flat buffer after it has been moved (i.e. to another maschine)
  /// It sets  mFlatBufferPtr to actualFlatBufferPtr.
  /// A daughter class should later update all the pointers inside the buffer in the new location.
  ///
  void setActualBufferAddress( char* actualFlatBufferPtr );

  
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
  void setFutureBufferAddress( char* futureFlatBufferPtr );


  /// _______________  Utilities  _______________________________________________

 public:

  /// Gives size of the flat buffer
  size_t getFlatBufferSize() const {return mFlatBufferSize;}
  
  /// Gives pointer to the flat buffer
  const char* getFlatBufferPtr() const {return mFlatBufferPtr;}

  /// Tells if the object is constructed
  bool isConstructed() const { 
    return (mConstructionMask & (unsigned int) ConstructionState::Constructed); 
  }

  /// Tells if the buffer is internal
  bool isBufferInternal() const { 
    return ( (mFlatBufferPtr!=nullptr) && (mFlatBufferPtr == mFlatBufferContainer.get()) ); 
  }



  /// _______________  Generic utilities  _______________________________________________

 public:

  /// Increases given size to achieve required alignment
  static size_t alignSize( size_t sizeBytes, size_t alignmentBytes )
  {
    return sizeBytes + (alignmentBytes - sizeBytes % alignmentBytes);
  }

  /// Relocates a pointer inside a buffer to the new buffer address
  template<class T>
    static T* relocatePointer( const char *oldBase, char *newBase, const T* ptr){ 
    return reinterpret_cast<T*>( newBase + (reinterpret_cast<const char*>(ptr) - oldBase) ); 
  }


 protected:   

  /// _______________  Data members  _______________________________________________


  /// Enumeration of construction states
  enum  ConstructionState : unsigned int { 
    NotConstructed = 0x0,    ///< the object is not constructed
    Constructed    = 0x1,    ///< the object is constructed, temporary memory is released
    InProgress     = 0x2     ///< construction started: temporary  memory is reserved
   };

  size_t mFlatBufferSize ;                      ///< Size of the flat buffer
  std::unique_ptr<char[]> mFlatBufferContainer; ///< Optional container for the flat buffer
  char* mFlatBufferPtr;                         ///< Pointer to the flat buffer    
  unsigned int mConstructionMask;               ///< mask for constructed object members, first two bytes are used by this class
};





/// ========================================================================================================
///
///       Inline implementations of methods
///
/// ========================================================================================================
 

inline FlatObject::FlatObject()
  :
  mFlatBufferSize( 0 ),
  mFlatBufferContainer( nullptr ),
  mFlatBufferPtr( nullptr ),
  mConstructionMask( ConstructionState::NotConstructed )
{  
  // Default Constructor: creates an empty uninitialized object
}


inline void FlatObject::startConstruction()
{ 
  /// Starts the construction procedure. A daughter class should reserve temporary memory.
  destroy();
  mConstructionMask =  ConstructionState::InProgress;
}
 

inline void FlatObject::destroy()
{
  /// Set the object to NotConstructed state, release the buffer
  mFlatBufferSize = 0; 
  mFlatBufferContainer.reset();
  mFlatBufferPtr = nullptr;
  mConstructionMask = ConstructionState::NotConstructed;
}


inline void FlatObject::finishConstruction( int flatBufferSize )
{
  /// Finishes construction: creates internal flat buffer. 
  /// A daughter class should put all created variable-size members to this buffer    

  assert( mConstructionMask & (unsigned int) ConstructionState::InProgress );

  mFlatBufferSize = flatBufferSize;
  mFlatBufferContainer.reset( new char[ mFlatBufferSize ] );
  mFlatBufferPtr = mFlatBufferContainer.get();

  memset( (void*)mFlatBufferPtr, 0, mFlatBufferSize ); // just to avoid random behavior in case of bugs

  mConstructionMask = (unsigned int) ConstructionState::Constructed; // clear other possible construction flags
}

      
inline void FlatObject::cloneFromObject( const FlatObject &obj, char *newFlatBufferPtr )
{

  /// Initializes from another object, copies data to newBufferPtr
  /// When newBufferPtr==nullptr, the internal container will be created, the data will be copied there. 
  /// obj can be *this

  assert( obj.isConstructed() );
  
  char *oldFlatBufferPtr = obj.mFlatBufferPtr;

  if( (newFlatBufferPtr==nullptr) || (newFlatBufferPtr == mFlatBufferContainer.get()) ){
    mFlatBufferContainer.reset( new char[ obj.mFlatBufferSize ] );
    newFlatBufferPtr = mFlatBufferContainer.get();
  } else {
    mFlatBufferContainer.reset();
  }
  std::memcpy( (void*) newFlatBufferPtr, (void*) oldFlatBufferPtr, obj.mFlatBufferSize );
 
  mFlatBufferSize = obj.mFlatBufferSize;
  mFlatBufferPtr = newFlatBufferPtr;
  mConstructionMask = (unsigned int) ConstructionState::Constructed; 
}
   

inline std::unique_ptr<char[]> FlatObject::releaseInternalBuffer()
{
  // returns an unique pointer to the internal buffer with all the rights. Makes the internal container variable empty.
  return std::move(mFlatBufferContainer); // must also work without move()
}


inline void FlatObject::moveBufferTo( char *newFlatBufferPtr )
{
  /// sets buffer pointer to the new address, move the buffer content there.
  if( newFlatBufferPtr==mFlatBufferContainer.get() ) return;
  if( newFlatBufferPtr==nullptr ){
    mFlatBufferContainer.reset( new char[ mFlatBufferSize ] );
    newFlatBufferPtr = mFlatBufferContainer.get();

  }
  std::memcpy( (void*) newFlatBufferPtr, (void*) mFlatBufferPtr, mFlatBufferSize );
  mFlatBufferPtr = newFlatBufferPtr;
  if( mFlatBufferPtr!=mFlatBufferContainer.get() ){
    mFlatBufferContainer.reset();
  }
}


  
inline void FlatObject::setActualBufferAddress( char* actualFlatBufferPtr )
{
  /// Sets the actual location of the external flat buffer after it has been moved (i.e. to another maschine)
  /// 
  /// It sets  mFlatBufferPtr to actualFlatBufferPtr.
  /// A daughter class should update all the pointers inside the buffer in the new location.
 
  assert( !isBufferInternal() );
  mFlatBufferPtr = actualFlatBufferPtr;
  mFlatBufferContainer.reset(); // for a case..
}


inline void FlatObject::setFutureBufferAddress( char* futureFlatBufferPtr )
{
  /// Sets a future location of the external flat buffer before moving it to this location.
  /// 
  /// A daughter class should already reset all the pointers inside the current buffer to the future location
  /// without touching memory in the future location.
 
  assert( !isBufferInternal() );
  mFlatBufferPtr = futureFlatBufferPtr;
  mFlatBufferContainer.reset(); // for a case..
}



}// namespace
}// namespace

#endif
