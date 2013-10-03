//-*- Mode: C++ -*-
// $Id: AliHLTArray.h 42754 2010-08-06 21:53:10Z aszostak $

// ****************************************************************************
// * This file is property of and copyright by the ALICE HLT Project          *
// * ALICE Experiment at CERN, All rights reserved.                           *
// *                                                                          *
// * Copyright (C) 2009 Matthias Kretz <kretz@kde.org>                        *
// *               for The ALICE HLT Project.                                 *
// *                                                                          *
// * Permission to use, copy, modify and distribute this software and its     *
// * documentation strictly for non-commercial purposes is hereby granted     *
// * without fee, provided that the above copyright notice appears in all     *
// * copies and that both the copyright notice and this permission notice     *
// * appear in the supporting documentation. The authors make no claims       *
// * about the suitability of this software for any purpose. It is            *
// * provided "as is" without express or implied warranty.                    *
// ****************************************************************************

/**
 * \file AliHLTArray.h
 * \author Matthias Kretz <kretz@kde.org>
 *
 * This file contains the classes AliHLTResizableArray and AliHLTFixedArray with AliHLTArray as base
 * class. It's a drop-in replacement for C-Arrays. It makes it easy to use variable sized arrays on
 * the stack and pass arrays as arguments to other functions with an optional bounds-checking
 * enabled for the whole time.
 */

#ifndef ALIHLTARRAY_H
#define ALIHLTARRAY_H

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)

#ifndef assert
#include <assert.h>
#endif

#if (defined(__MMX__) || defined(__SSE__))
#if defined(__GNUC__)
#if __GNUC__ > 3
#define USE_MM_MALLOC
#endif
#else // not gcc, assume it can use _mm_malloc since it supports MMX/SSE
#define USE_MM_MALLOC
#endif
#endif

#ifdef USE_MM_MALLOC
#undef USE_MM_MALLOC
#endif

#ifdef USE_MM_MALLOC
#include <mm_malloc.h>
#else
#include <cstdlib>
#endif

enum {
  AliHLTFullyCacheLineAligned = -1
};

#if defined(__CUDACC__) & 0
#define ALIHLTARRAY_STATIC_ASSERT(a, b)
#define ALIHLTARRAY_STATIC_ASSERT_NC(a, b)
#else
namespace AliHLTArrayInternal
{
  template<bool> class STATIC_ASSERT_FAILURE;
  template<> class STATIC_ASSERT_FAILURE<true> {};
}

#define ALIHLTARRAY_STATIC_ASSERT_CONCAT_HELPER(a, b) a##b
#define ALIHLTARRAY_STATIC_ASSERT_CONCAT(a, b) ALIHLTARRAY_STATIC_ASSERT_CONCAT_HELPER(a, b)
#define ALIHLTARRAY_STATIC_ASSERT_NC(cond, msg) \
  typedef AliHLTArrayInternal::STATIC_ASSERT_FAILURE<cond> ALIHLTARRAY_STATIC_ASSERT_CONCAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__); \
  ALIHLTARRAY_STATIC_ASSERT_CONCAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__) Error_##msg
#define ALIHLTARRAY_STATIC_ASSERT(cond, msg) ALIHLTARRAY_STATIC_ASSERT_NC(cond, msg); (void) Error_##msg
#endif

template<typename T, int Dim> class AliHLTArray;

namespace AliHLTInternal
{
  template<unsigned int Size> struct Padding { char fPadding[Size]; };
  template<> struct Padding<0> {};
  template<typename T> struct CacheLineSizeHelperData { T fData; };
  template<typename T> struct CacheLineSizeHelperEnums {
    enum {
      CacheLineSize = 64,
      MaskedSize = sizeof( T ) & ( CacheLineSize - 1 ),
      RequiredSize = MaskedSize == 0 ? sizeof( T ) : sizeof( T ) + CacheLineSize - MaskedSize,
      PaddingSize = RequiredSize - sizeof( T )
    };
  };
  template<typename T> class CacheLineSizeHelper : private CacheLineSizeHelperData<T>, private Padding<CacheLineSizeHelperEnums<T>::PaddingSize>
  {
    public:
      operator T &() { return CacheLineSizeHelperData<T>::fData; }
      operator const T &() const { return CacheLineSizeHelperData<T>::fData; }
      //const T &operator=( const T &rhs ) { CacheLineSizeHelperData<T>::fData = rhs; }

    private:
  };
  template<typename T, int alignment> struct TypeForAlignmentHelper { typedef T Type; };
  template<typename T> struct TypeForAlignmentHelper<T, AliHLTFullyCacheLineAligned> { typedef CacheLineSizeHelper<T> Type; };

  // XXX
  // The ArrayBoundsCheck and Allocator classes implement a virtual destructor only in order to
  // silence the -Weffc++ warning. It really is not required for these classes to have a virtual
  // dtor since polymorphism is not used (AliHLTResizableArray and AliHLTFixedArray are allocated on
  // the stack only). The virtual dtor only adds an unnecessary vtable to the code.
#ifndef ENABLE_ARRAY_BOUNDS_CHECKING
  /**
   * no-op implementation that for no-bounds-checking
   */
  class ArrayBoundsCheck
  {
    protected:
      virtual inline ~ArrayBoundsCheck() {}
      inline bool IsInBounds( int ) const { return true; }
      inline void SetBounds( int, int ) {}
      inline void MoveBounds( int ) {}
      inline void ReinterpretCast( const ArrayBoundsCheck &, int, int ) {}
  };
#define BOUNDS_CHECK(x, y)
#else
  /**
   * implementation for bounds-checking.
   */
  class ArrayBoundsCheck
  {
    protected:
      virtual inline ~ArrayBoundsCheck() {}
      /**
       * checks whether the given offset is valid
       */
      inline bool IsInBounds( int x ) const;
      /**
       * set the start and end offsets that are still valid
       */
      inline void SetBounds( int start, int end ) { fStart = start; fEnd = end; }
      /**
       * move the start and end offsets by the same amount
       */
      inline void MoveBounds( int d ) { fStart += d; fEnd += d; }

      inline void ReinterpretCast( const ArrayBoundsCheck &other, int sizeofOld, int sizeofNew ) {
        fStart = other.fStart * sizeofNew / sizeofOld;
        fEnd = other.fEnd * sizeofNew / sizeofOld;
      }

    private:
      int fStart; // start
      int fEnd;   // end
  };
#define BOUNDS_CHECK(x, y) if (AliHLTInternal::ArrayBoundsCheck::IsInBounds(x)) {} else return y
#endif
  template<typename T, int alignment> class Allocator
  {
    public:
#ifdef USE_MM_MALLOC
      static inline T *Alloc( int s ) { T *p = reinterpret_cast<T *>( _mm_malloc( s * sizeof( T ), alignment ) ); return new( p ) T[s]; }
      static inline void Free( T *const p, int size ) {
        for ( int i = 0; i < size; ++i ) {
          p[i].~T();
        }
        _mm_free( p );
      }
#else
      static inline T *Alloc( int s ) { T *p; posix_memalign( &p, alignment, s * sizeof( T ) ); return new( p ) T[s]; }
      static inline void Free( T *const p, int size ) {
        for ( int i = 0; i < size; ++i ) {
          p[i].~T();
        }
        std::free( p );
      }
#endif
  };
  template<typename T> class Allocator<T, AliHLTFullyCacheLineAligned>
  {
    public:
      typedef CacheLineSizeHelper<T> T2;
#ifdef USE_MM_MALLOC
      static inline T2 *Alloc( int s ) { T2 *p = reinterpret_cast<T2 *>( _mm_malloc( s * sizeof( T2 ), 128 ) ); return new( p ) T2[s]; }
      static inline void Free( T2 *const p, int size ) {
        for ( int i = 0; i < size; ++i ) {
			p[i].~T2();
        }
        _mm_free( p );
      }
#else
      static inline T2 *Alloc( int s ) { T2 *p; posix_memalign( &p, 128, s * sizeof( T2 ) ); return new( p ) T2[s]; }
      static inline void Free( T2 *const p, int size ) {
        for ( int i = 0; i < size; ++i ) {
			p[i].~T2();
        }
        std::free( p );
      }
#endif
  };
  template<typename T> class Allocator<T, 0>
  {
    public:
      static inline T *Alloc( int s ) { return new T[s]; }
      static inline void Free( const T *const p, int ) { delete[] p; }
  };

  template<typename T> struct ReturnTypeHelper { typedef T Type; };
  template<typename T> struct ReturnTypeHelper<CacheLineSizeHelper<T> > { typedef T Type; };
  /**
   * Array base class for dimension dependent behavior
   */
  template<typename T, int Dim> class ArrayBase;

  /**
   * 1-dim arrays only have operator[]
   */
  template<typename T>
  class ArrayBase<T, 1> : public ArrayBoundsCheck
  {
      friend class ArrayBase<T, 2>;
    public:
      ArrayBase() : fData( 0 ), fSize( 0 ) {} // XXX really shouldn't be done. But -Weffc++ wants it so
      ArrayBase( const ArrayBase &rhs ) : ArrayBoundsCheck( rhs ), fData( rhs.fData ), fSize( rhs.fSize ) {} // XXX
      ArrayBase &operator=( const ArrayBase &rhs ) { ArrayBoundsCheck::operator=( rhs ); fData = rhs.fData; return *this; } // XXX
      typedef typename ReturnTypeHelper<T>::Type R;
      /**
       * return a reference to the value at the given index
       */
      inline R &operator[]( int x ) { BOUNDS_CHECK( x, fData[0] ); return fData[x]; }
      /**
       * return a const reference to the value at the given index
       */
      inline const R &operator[]( int x ) const { BOUNDS_CHECK( x, fData[0] ); return fData[x]; }

    protected:
      T *fData;  // actual data
      int fSize; // data size
      inline void SetSize( int x, int, int ) { fSize = x; }
  };

  /**
   * 2-dim arrays should use operator(int, int)
   * operator[] can be used to return a 1-dim array
   */
  template<typename T>
  class ArrayBase<T, 2> : public ArrayBoundsCheck
  {
      friend class ArrayBase<T, 3>;
    public:
      ArrayBase() : fData( 0 ), fSize( 0 ), fStride( 0 ) {} // XXX really shouldn't be done. But -Weffc++ wants it so
      ArrayBase( const ArrayBase &rhs ) : ArrayBoundsCheck( rhs ), fData( rhs.fData ), fSize( rhs.fSize ), fStride( rhs.fStride ) {} // XXX
      ArrayBase &operator=( const ArrayBase &rhs ) { ArrayBoundsCheck::operator=( rhs ); fData = rhs.fData; fSize = rhs.fSize; fStride = rhs.fStride; return *this; } // XXX
      typedef typename ReturnTypeHelper<T>::Type R;
      /**
       * return a reference to the value at the given indexes
       */
      inline R &operator()( int x, int y ) { BOUNDS_CHECK( x * fStride + y, fData[0] ); return fData[x * fStride + y]; }
      /**
       * return a const reference to the value at the given indexes
       */
      inline const R &operator()( int x, int y ) const { BOUNDS_CHECK( x * fStride + y, fData[0] ); return fData[x * fStride + y]; }
      /**
       * return a 1-dim array at the given index. This makes it behave like a 2-dim C-Array.
       */
      inline AliHLTArray<T, 1> operator[]( int x );
      /**
       * return a const 1-dim array at the given index. This makes it behave like a 2-dim C-Array.
       */
      inline const AliHLTArray<T, 1> operator[]( int x ) const;

    protected:
      T *fData;    // actual data
      int fSize;   // data size
      int fStride; // 
      inline void SetSize( int x, int y, int ) { fStride = y; fSize = x * y; }
  };

  /**
   * 3-dim arrays should use operator(int, int, int)
   * operator[] can be used to return a 2-dim array
   */
  template<typename T>
  class ArrayBase<T, 3> : public ArrayBoundsCheck
  {
    public:
      ArrayBase() : fData( 0 ), fSize( 0 ), fStrideX( 0 ), fStrideY( 0 ) {} // XXX really shouldn't be done. But -Weffc++ wants it so
      ArrayBase( const ArrayBase &rhs ) : ArrayBoundsCheck( rhs ), fData( rhs.fData ), fSize( rhs.fSize ), fStrideX( rhs.fStrideX ), fStrideY( rhs.fStrideY ) {} // XXX
      ArrayBase &operator=( const ArrayBase &rhs ) { ArrayBoundsCheck::operator=( rhs ); fData = rhs.fData; fSize = rhs.fSize; fStrideX = rhs.fStrideX; fStrideY = rhs.fStrideY; return *this; } // XXX
      // Stopped working on GCC 4.5.0
      //typedef typename ReturnTypeHelper<T>::Type R;
      /**
       * return a reference to the value at the given indexes
       */
      inline typename ReturnTypeHelper<T>::Type &operator()( int x, int y, int z );
      /**
       * return a const reference to the value at the given indexes
       */
      inline const typename ReturnTypeHelper<T>::Type &operator()( int x, int y, int z ) const;
      /**
       * return a 2-dim array at the given index. This makes it behave like a 3-dim C-Array.
       */
      inline AliHLTArray<T, 2> operator[]( int x );
      /**
       * return a const 2-dim array at the given index. This makes it behave like a 3-dim C-Array.
       */
      inline const AliHLTArray<T, 2> operator[]( int x ) const;

    protected:
      T *fData;     // actual data
      int fSize;    // data size
      int fStrideX; //
      int fStrideY; //
      inline void SetSize( int x, int y, int z ) { fStrideX = y * z; fStrideY = z; fSize = fStrideX * x; }
  };

  template<typename T, unsigned int Size, int _alignment> class AlignedData
  {
    public:
      T *ConstructAlignedData() {
        const int offset = reinterpret_cast<unsigned long>( &fUnalignedArray[0] ) & ( Alignment - 1 );
        void *mem = &fUnalignedArray[0] + ( Alignment - offset );
        return new( mem ) T[Size];
      }
      ~AlignedData() {
        const int offset = reinterpret_cast<unsigned long>( &fUnalignedArray[0] ) & ( Alignment - 1 );
        T *mem = reinterpret_cast<T *>( &fUnalignedArray[0] + ( Alignment - offset ) );
        for ( unsigned int i = 0; i < Size; ++i ) {
          mem[i].~T();
        }
      }
    private:
      enum {
        Alignment = _alignment == AliHLTFullyCacheLineAligned ? 128 : _alignment,
        PaddedSize = Size * sizeof( T ) + Alignment
      };
      ALIHLTARRAY_STATIC_ASSERT_NC( ( Alignment & ( Alignment - 1 ) ) == 0, alignment_needs_to_be_a_multiple_of_2 );

      char fUnalignedArray[PaddedSize]; //
  };
  template<typename T, unsigned int Size> class AlignedData<T, Size, 0>
  {
    public:
      T *ConstructAlignedData() { return &fArray[0]; }
    private:
      T fArray[Size]; //
  };
} // namespace AliHLTInternal

/**
 * C-Array like class with the dimension dependent behavior defined in the ArrayBase class
 */
template < typename T, int Dim = 1 >
class AliHLTArray : public AliHLTInternal::ArrayBase<T, Dim>
{
  public:
    typedef AliHLTInternal::ArrayBase<T, Dim> Parent;

    /**
     * Returns the number of elements in the array. If it is a multi-dimensional array the size is
     * the multiplication of the dimensions ( e.g. a 10 x 20 array returns 200 as its size ).
     */
    inline int Size() const { return Parent::fSize; }

    /**
     * allows you to check for validity of the array by casting to bool
     */
    inline operator bool() const { return Parent::fData != 0; }
    /**
     * allows you to check for validity of the array
     */
    inline bool IsValid() const { return Parent::fData != 0; }

    /**
     * returns a reference to the data at index 0
     */
    inline T &operator*() { BOUNDS_CHECK( 0, Parent::fData[0] ); return *Parent::fData; }
    /**
     * returns a const reference to the data at index 0
     */
    inline const T &operator*() const { BOUNDS_CHECK( 0, Parent::fData[0] ); return *Parent::fData; }

    /**
     * returns a pointer to the data
     * This circumvents bounds checking so it should not be used.
     */
    inline T *Data() { return Parent::fData; }
    /**
     * returns a const pointer to the data
     * This circumvents bounds checking so it should not be used.
     */
    inline const T *Data() const { return Parent::fData; }

    /**
     * moves the array base pointer so that the data that was once at index 0 will then be at index -x
     */
    inline AliHLTArray operator+( int x ) const;
    /**
     * moves the array base pointer so that the data that was once at index 0 will then be at index x
     */
    inline AliHLTArray operator-( int x ) const;

#ifndef HLTCA_GPUCODE
    template<typename Other> inline AliHLTArray<Other, Dim> ReinterpretCast() const {
      AliHLTArray<Other, Dim> r;
      r.fData = reinterpret_cast<Other *>( Parent::fData );
      r.ReinterpretCast( *this, sizeof( T ), sizeof( Other ) );
    }
#endif
};

/**
 * Owns the data. When it goes out of scope the data is freed.
 *
 * The memory is allocated on the heap.
 *
 * Instantiate this class on the stack. Allocation on the heap is disallowed.
 *
 * \param T type of the entries in the array.
 * \param Dim selects the operator[]/operator() behavior it should have. I.e. makes it behave like a
 * 1-, 2- or 3-dim array. (defaults to 1)
 * \param alignment Defaults to 0 (default alignment). Other valid values are any multiples of 2.
 *                  This is especially useful for aligning data for SIMD vectors.
 *
 * \warning when using alignment the type T may not have a destructor (well it may, but it won't be
 * called)
 *
 * Example:
 * \code
 * void init( AliHLTArray<int> a, int size )
 * {
 *   for ( int i = 0; i < size; ++i ) {
 *     a[i] = i;
 *   }
 * }
 *
 * int size = ...;
 * AliHLTResizableArray<int> foo( size ); // notice that size doesn't have to be a constant like it
 *                                        // has to be for C-Arrays in ISO C++
 * init( foo, size );
 * // now foo[i] == i
 *
 * \endcode
 */
template < typename T, int Dim = 1, int alignment = 0 >
class AliHLTResizableArray : public AliHLTArray<typename AliHLTInternal::TypeForAlignmentHelper<T, alignment>::Type, Dim>
{
  public:
    typedef typename AliHLTInternal::TypeForAlignmentHelper<T, alignment>::Type T2;
    typedef AliHLTInternal::ArrayBase<T2, Dim> Parent;
    /**
     * does not allocate any memory
     */
    inline AliHLTResizableArray();
    /**
     * use for 1-dim arrays: allocates x * sizeof(T) bytes for the array
     */
    inline AliHLTResizableArray( int x );
    /**
     * use for 2-dim arrays: allocates x * y * sizeof(T) bytes for the array
     */
    inline AliHLTResizableArray( int x, int y );
    /**
     * use for 3-dim arrays: allocates x * y * z * sizeof(T) bytes for the array
     */
    inline AliHLTResizableArray( int x, int y, int z );

    /**
     * frees the data
     */
    inline ~AliHLTResizableArray() { AliHLTInternal::Allocator<T, alignment>::Free( Parent::fData, Parent::fSize ); }

    /**
     * use for 1-dim arrays: resizes the memory for the array to x * sizeof(T) bytes.
     *
     * \warning this does not keep your previous data. If you were looking for this you probably
     * want to use std::vector instead.
     */
    inline void Resize( int x );
    /**
     * use for 2-dim arrays: resizes the memory for the array to x * y * sizeof(T) bytes.
     *
     * \warning this does not keep your previous data. If you were looking for this you probably
     * want to use std::vector instead.
     */
    inline void Resize( int x, int y );
    /**
     * use for 3-dim arrays: resizes the memory for the array to x * y * z * sizeof(T) bytes.
     *
     * \warning this does not keep your previous data. If you were looking for this you probably
     * want to use std::vector instead.
     */
    inline void Resize( int x, int y, int z );

  private:
    // disable allocation on the heap
    void *operator new( size_t );

    // disable copy
    AliHLTResizableArray( const AliHLTResizableArray & );
    AliHLTResizableArray &operator=( const AliHLTResizableArray & );
};

template < unsigned int x, unsigned int y = 0, unsigned int z = 0 > class AliHLTArraySize
{
  public:
    enum {
      Size = y == 0 ? x : ( z == 0 ? x * y : x * y * z ),
      Dim = y == 0 ? 1 : ( z == 0 ? 2 : 3 ),
      X = x, Y = y, Z = z
    };
};

/**
 * Owns the data. When it goes out of scope the data is freed.
 *
 * The memory is allocated on the stack.
 *
 * Instantiate this class on the stack.
 *
 * \param T type of the entries in the array.
 * \param Size number of entries in the array.
 * \param Dim selects the operator[]/operator() behavior it should have. I.e. makes it behave like a
 * 1-, 2- or 3-dim array. (defaults to 1)
 */
template < typename T, typename Size, int alignment = 0 >
class AliHLTFixedArray : public AliHLTArray<typename AliHLTInternal::TypeForAlignmentHelper<T, alignment>::Type, Size::Dim>
{
  public:
    typedef typename AliHLTInternal::TypeForAlignmentHelper<T, alignment>::Type T2;
    typedef AliHLTInternal::ArrayBase<T2, Size::Dim> Parent;
    inline AliHLTFixedArray() {
      Parent::fData = fFixedArray.ConstructAlignedData();
      Parent::SetBounds( 0, Size::Size - 1 );
      SetSize( Size::X, Size::Y, Size::Z );
    }

  private:
    AliHLTInternal::AlignedData<typename AliHLTInternal::TypeForAlignmentHelper<T, alignment>::Type, Size::Size, alignment> fFixedArray; //

    // disable allocation on the heap
    void *operator new( size_t );

    // disable copy
#ifdef HLTCA_GPUCODE
#else
    AliHLTFixedArray( const AliHLTFixedArray & );
    AliHLTFixedArray &operator=( const AliHLTFixedArray & );
#endif
};




////////////////////////
//// implementation ////
////////////////////////




namespace AliHLTInternal
{
#ifdef ENABLE_ARRAY_BOUNDS_CHECKING
  inline bool ArrayBoundsCheck::IsInBounds( int x ) const
  {
    assert( x >= fStart );
    assert( x <= fEnd );
    return ( x >= fStart && x <= fEnd );
  }
#endif

  template<typename T>
  inline AliHLTArray<T, 1> ArrayBase<T, 2>::operator[]( int x )
  {
    x *= fStride;
    typedef AliHLTArray<T, 1> AT1;
    BOUNDS_CHECK( x, AT1() );
    AliHLTArray<T, 1> a;
    a.fData = &fData[x];
    a.ArrayBoundsCheck::operator=( *this );
    a.MoveBounds( -x );
    return a;
  }

  template<typename T>
  inline const AliHLTArray<T, 1> ArrayBase<T, 2>::operator[]( int x ) const
  {
    x *= fStride;
    typedef AliHLTArray<T, 1> AT1;
    BOUNDS_CHECK( x, AT1() );
    AliHLTArray<T, 1> a;
    a.fData = &fData[x];
    a.ArrayBoundsCheck::operator=( *this );
    a.MoveBounds( -x );
    return a;
  }

  template<typename T>
  inline typename AliHLTInternal::ReturnTypeHelper<T>::Type &ArrayBase<T, 3>::operator()( int x, int y, int z )
  {
    BOUNDS_CHECK( x * fStrideX + y + fStrideY + z, fData[0] );
    return fData[x * fStrideX + y + fStrideY + z];
  }
  template<typename T>
  inline const typename AliHLTInternal::ReturnTypeHelper<T>::Type &ArrayBase<T, 3>::operator()( int x, int y, int z ) const
  {
    BOUNDS_CHECK( x * fStrideX + y + fStrideY + z, fData[0] );
    return fData[x * fStrideX + y + fStrideY + z];
  }
  template<typename T>
  inline AliHLTArray<T, 2> ArrayBase<T, 3>::operator[]( int x )
  {
    x *= fStrideX;
    typedef AliHLTArray<T, 2> AT2;
    BOUNDS_CHECK( x, AT2() );
    AliHLTArray<T, 2> a;
    a.fData = &fData[x];
    a.fStride = fStrideY;
    a.ArrayBoundsCheck::operator=( *this );
    a.MoveBounds( -x );
    return a;
  }
  template<typename T>
  inline const AliHLTArray<T, 2> ArrayBase<T, 3>::operator[]( int x ) const
  {
    x *= fStrideX;
    typedef AliHLTArray<T, 2> AT2;
    BOUNDS_CHECK( x, AT2() );
    AliHLTArray<T, 2> a;
    a.fData = &fData[x];
    a.fStride = fStrideY;
    a.ArrayBoundsCheck::operator=( *this );
    a.MoveBounds( -x );
    return a;
  }
} // namespace AliHLTInternal


template<typename T, int Dim>
inline AliHLTArray<T, Dim> AliHLTArray<T, Dim>::operator+( int x ) const
{
  AliHLTArray<T, Dim> r( *this );
  r.fData += x;
  r.MoveBounds( -x );
  return r;
}
template<typename T, int Dim>
inline AliHLTArray<T, Dim> AliHLTArray<T, Dim>::operator-( int x ) const
{
  AliHLTArray<T, Dim> r( *this );
  r.fData -= x;
  r.MoveBounds( x );
  return r;
}

template<typename T, int Dim, int alignment>
inline AliHLTResizableArray<T, Dim, alignment>::AliHLTResizableArray()
{
  Parent::fData = 0;
  Parent::SetSize( 0, 0, 0 );
  Parent::SetBounds( 0, -1 );
}
template<typename T, int Dim, int alignment>
inline AliHLTResizableArray<T, Dim, alignment>::AliHLTResizableArray( int x )
{
  ALIHLTARRAY_STATIC_ASSERT( Dim == 1, AliHLTResizableArray1_used_with_incorrect_dimension );
  Parent::fData = AliHLTInternal::Allocator<T, alignment>::Alloc( x );
  Parent::SetSize( x, 0, 0 );
  Parent::SetBounds( 0, x - 1 );
}
template<typename T, int Dim, int alignment>
inline AliHLTResizableArray<T, Dim, alignment>::AliHLTResizableArray( int x, int y )
{
  ALIHLTARRAY_STATIC_ASSERT( Dim == 2, AliHLTResizableArray2_used_with_incorrect_dimension );
  Parent::fData = AliHLTInternal::Allocator<T, alignment>::Alloc( x * y );
  Parent::SetSize( x, y, 0 );
  Parent::SetBounds( 0, x * y - 1 );
}
template<typename T, int Dim, int alignment>
inline AliHLTResizableArray<T, Dim, alignment>::AliHLTResizableArray( int x, int y, int z )
{
  ALIHLTARRAY_STATIC_ASSERT( Dim == 3, AliHLTResizableArray3_used_with_incorrect_dimension );
  Parent::fData = AliHLTInternal::Allocator<T, alignment>::Alloc( x * y * z );
  Parent::SetSize( x, y, z );
  Parent::SetBounds( 0, x * y * z - 1 );
}
template<typename T, int Dim, int alignment>
inline void AliHLTResizableArray<T, Dim, alignment>::Resize( int x )
{
  ALIHLTARRAY_STATIC_ASSERT( Dim == 1, AliHLTResizableArray1_resize_used_with_incorrect_dimension );
  AliHLTInternal::Allocator<T, alignment>::Free( Parent::fData, Parent::fSize );
  Parent::fData = ( x == 0 ) ? 0 : AliHLTInternal::Allocator<T, alignment>::Alloc( x );
  Parent::SetSize( x, 0, 0 );
  Parent::SetBounds( 0, x - 1 );
}
template<typename T, int Dim, int alignment>
inline void AliHLTResizableArray<T, Dim, alignment>::Resize( int x, int y )
{
  ALIHLTARRAY_STATIC_ASSERT( Dim == 2, AliHLTResizableArray2_resize_used_with_incorrect_dimension );
  AliHLTInternal::Allocator<T, alignment>::Free( Parent::fData, Parent::fSize );
  Parent::fData = ( x == 0 ) ? 0 : AliHLTInternal::Allocator<T, alignment>::Alloc( x * y );
  Parent::SetSize( x, y, 0 );
  Parent::SetBounds( 0, x * y - 1 );
}
template<typename T, int Dim, int alignment>
inline void AliHLTResizableArray<T, Dim, alignment>::Resize( int x, int y, int z )
{
  ALIHLTARRAY_STATIC_ASSERT( Dim == 3, AliHLTResizableArray3_resize_used_with_incorrect_dimension );
  AliHLTInternal::Allocator<T, alignment>::Free( Parent::fData, Parent::fSize );
  Parent::fData = ( x == 0 ) ? 0 : AliHLTInternal::Allocator<T, alignment>::Alloc( x * y * z );
  Parent::SetSize( x, y, z );
  Parent::SetBounds( 0, x * y * z - 1 );
}

#undef BOUNDS_CHECK

#endif

#endif // ALIHLTARRAY_H
