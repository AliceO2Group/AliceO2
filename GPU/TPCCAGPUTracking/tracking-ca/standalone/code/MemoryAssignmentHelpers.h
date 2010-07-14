// **************************************************************************
// * This file is property of and copyright by the ALICE HLT Project        *
// * All rights reserved.                                                   *
// *                                                                        *
// * Primary Authors:                                                       *
// *     Copyright 2009       Matthias Kretz <kretz@kde.org>                *
// *                                                                        *
// * Permission to use, copy, modify and distribute this software and its   *
// * documentation strictly for non-commercial purposes is hereby granted   *
// * without fee, provided that the above copyright notice appears in all   *
// * copies and that both the copyright notice and this permission notice   *
// * appear in the supporting documentation. The authors make no claims     *
// * about the suitability of this software for any purpose. It is          *
// * provided "as is" without express or implied warranty.                  *
// **************************************************************************

#ifndef MEMORYASSIGNMENTHELPERS_H
#define MEMORYASSIGNMENTHELPERS_H

#ifndef assert
#include <assert.h>
#endif //!assert

template<unsigned int X>
GPUhd() static inline void AlignTo( char *&mem )
{
  STATIC_ASSERT( ( X & ( X - 1 ) ) == 0, X_needs_to_be_a_multiple_of_2 );
  const int offset = reinterpret_cast<unsigned long long>( mem ) & ( X - 1 );
  if ( offset > 0 ) {
    mem += ( X - offset );
  }
  //assert( ( reinterpret_cast<unsigned long>( mem ) & ( X - 1 ) ) == 0 );
}

template<unsigned int X>
GPUhd() static inline unsigned int NextMultipleOf( unsigned int value )
{
  STATIC_ASSERT( ( X & ( X - 1 ) ) == 0, X_needs_to_be_a_multiple_of_2 );
  const int offset = value & ( X - 1 );
  if ( offset > 0 ) {
    return value + X - offset;
  }
  return value;
}

template<typename T, unsigned int Alignment> static T *AssignMemory( char *&mem, unsigned int size )
{
  STATIC_ASSERT( ( Alignment & ( Alignment - 1 ) ) == 0, Alignment_needs_to_be_a_multiple_of_2 );
  AlignTo<Alignment> ( mem );
  T *r = reinterpret_cast<T *>( mem );
  mem += size * sizeof( T );
  return r;
}

template<typename T, unsigned int Alignment> static inline T *AssignMemory( char *&mem, unsigned int stride, unsigned int count )
{
  assert( 0 == ( stride & ( Alignment - 1 ) ) );
  return AssignMemory<T, Alignment>( mem, stride * count );
}

template<typename T, unsigned int Alignment> GPUhd() static T *_assignMemory( char *&mem, unsigned int size )
{
  STATIC_ASSERT( ( Alignment & ( Alignment - 1 ) ) == 0, Alignment_needs_to_be_a_multiple_of_2 );
  AlignTo<Alignment < sizeof( HLTCA_GPU_ROWALIGNMENT ) ? sizeof( HLTCA_GPU_ROWALIGNMENT ) : Alignment>( mem );
  T *r = reinterpret_cast<T *>( mem );
  mem += size * sizeof( T );
  return r;
}

template<typename T> GPUhd() static inline void AssignMemory( T *&dst, char *&mem, int count )
{
	dst = _assignMemory < T, ( sizeof( T ) & ( sizeof( T ) - 1 ) ) == 0 && sizeof( T ) <= 16 ? sizeof( T ) : sizeof( void * ) > ( mem, count );
}

#endif // MEMORYASSIGNMENTHELPERS_H
