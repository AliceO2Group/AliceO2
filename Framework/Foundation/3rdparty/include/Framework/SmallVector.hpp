/*
 * This is a standalone copy of the SmallVector class from LLVM with no
 * dependencies outside standard library.
 *
 * The original code is distributed under the University of Illinois/NCSA Open
 * Source License. See below for details, or visit
 * <http://releases.llvm.org/5.0.0/LICENSE.TXT>.
 *
 * The standalone copy was made by Ollie Etherington as is distributed under
 * the same license.
 */

/*
 * USAGE:
 * This is a 'single header library' in the same vein as Sean Barrett's STB
 * libraries <https://github.com/nothings/stb>.
 *
 * The C++ implementation is created in *one* file be doing:
 *    #define LLVM_SMALL_VECTOR_IMPLEMENTATION
 * before including this file.
 *
 * You can also optionally define LLVM_SMALL_VECTOR_BAD_ALLOC to be the name of
 * a function that takes a single const char * argument to be called in the
 * event that the vector runs out of memory. The argument is a string
 * containing the reason for the error. This only needs to be done in the
 * implementing file as it's a part of the implementation, not the header.
 *
 * The vector can then be used as follows:
 * static void fill(llvm::SmallVectorImpl<size_t> &v) {
 *     for (size_t i = 0; i < 16; i++) v.push_back(i);
 * }
 *
 * static void print(llvm::SmallVectorImpl<size_t> &v) {
 *     for (auto i : v)
 *         printf("%zu ", i);
 *     printf("\n");
 * }
 *
 * int main(int argc, char **argv) {
 *     llvm::SmallVector<size_t, 16> vec0;
 *     llvm::SmallVector<size_t, 32> vec1;
 *     fill(vec0);
 *     fill(vec1);
 *     print(vec0);
 *     print(vec1);
 *     return 0;
 * }
 */

/*
   ============================================================================
   LLVM Release License
   ============================================================================
   University of Illinois/NCSA
   Open Source License

   Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
   All rights reserved.

   Developed by:

   LLVM Team

   University of Illinois at Urbana-Champaign

   http://llvm.org

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimers.

     * Redistributions in binary form must reproduce the above copyright
	   notice, this list of conditions and the following disclaimers in the
       documentation and/or other materials provided with the distribution.

     * Neither the names of the LLVM Team, University of Illinois at
	   Urbana-Champaign, nor the names of its contributors may be used to
       endorse or promote products derived from this Software without specific
       prior written permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   WITH THE SOFTWARE.
 */

#ifndef LLVM_ADT_SMALLVECTOR_H
#define LLVM_ADT_SMALLVECTOR_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

/*
 * From llvm/Support/Compiler.h
 */
#ifndef LLVM_NODISCARD
#define LLVM_NODISCARD [[nodiscard]]
#endif

/// \macro LLVM_GNUC_PREREQ
/// \brief Extend the default __GNUC_PREREQ even if glibc's features.h isn't
/// available.
#ifndef LLVM_GNUC_PREREQ
# if defined(__GNUC__)&&defined(__GNUC_MINOR__)&&defined(__GNUC_PATCHLEVEL__)
#  define LLVM_GNUC_PREREQ(maj, min, patch) \
    ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) + __GNUC_PATCHLEVEL__ >= \
     ((maj) << 20) + ((min) << 10) + (patch))
# elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#  define LLVM_GNUC_PREREQ(maj, min, patch) \
    ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10)>=((maj) << 20) + ((min) << 10))
# else
#  define LLVM_GNUC_PREREQ(maj, min, patch) 0
# endif
#endif

/*
 * From llvm/Support/type_traits.h
 */
#ifndef __has_feature
#define LLVM_DEFINED_HAS_FEATURE
#define __has_feature(x) 0
#endif

/// \macro LLVM_ALIGNAS
/// \brief Used to specify a minimum alignment for a structure or variable.
#if __GNUC__ && !__has_feature(cxx_alignas) && !LLVM_GNUC_PREREQ(4, 8, 1)
# define LLVM_ALIGNAS(x) __attribute__((aligned(x)))
#else
# define LLVM_ALIGNAS(x) alignas(x)
#endif

#if __has_builtin(__builtin_expect) || LLVM_GNUC_PREREQ(4, 0, 0)
#define LLVM_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define LLVM_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define LLVM_LIKELY(EXPR) (EXPR)
#define LLVM_UNLIKELY(EXPR) (EXPR)
#endif

/// LLVM_ATTRIBUTE_ALWAYS_INLINE - On compilers where we have a directive to do
/// so, mark a method "always inline" because it is performance sensitive. GCC
/// 3.4 supported this but is buggy in various cases and produces unimplemented
/// errors, just use it in GCC 4.0 and later.
#if __has_attribute(always_inline) || LLVM_GNUC_PREREQ(4, 0, 0)
#define LLVM_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_ALWAYS_INLINE __forceinline
#else
#define LLVM_ATTRIBUTE_ALWAYS_INLINE
#endif

namespace llvm {

/*
 * From llvm/ADT/iterator_range.h
 */
/// \brief A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

public:
  //TODO: Add SFINAE to test that the Container's iterators match the range's
  //      iterators.
  template <typename Container>
  iterator_range(Container &&c)
  //TODO: Consider ADL/non-member begin/end calls.
      : begin_iterator(c.begin()), end_iterator(c.end()) {}
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
};

/// \brief Convenience function for iterating over sub-ranges.
///
/// This provides a bit of syntactic sugar to make using sub-ranges
/// in for loops a bit easier. Analogous to std::make_pair().
template <class T> iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T> iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}

template<typename T>
iterator_range<decltype(begin(std::declval<T>()))> drop_begin(T &&t, int n) {
  return make_range(std::next(begin(t), n), end(t));
}

/*
 * From llvm/Support/AlignOf.h
 */
/// \struct AlignedCharArray
/// \brief Helper for building an aligned character array type.
///
/// This template is used to explicitly build up a collection of aligned
/// character array types. We have to build these up using a macro and explicit
/// specialization to cope with MSVC (at least till 2015) where only an
/// integer literal can be used to specify an alignment constraint. Once built
/// up here, we can then begin to indirect between these using normal C++
/// template parameters.

// MSVC requires special handling here.
#ifndef _MSC_VER

template<std::size_t Alignment, std::size_t Size>
struct AlignedCharArray {
  LLVM_ALIGNAS(Alignment) char buffer[Size];
};

#else // _MSC_VER

/// \brief Create a type with an aligned char buffer.
template<std::size_t Alignment, std::size_t Size>
struct AlignedCharArray;

// We provide special variations of this template for the most common
// alignments because __declspec(align(...)) doesn't actually work when it is
// a member of a by-value function argument in MSVC, even if the alignment
// request is something reasonably like 8-byte or 16-byte. Note that we can't
// even include the declspec with the union that forces the alignment because
// MSVC warns on the existence of the declspec despite the union member forcing
// proper alignment.

template<std::size_t Size>
struct AlignedCharArray<1, Size> {
  union {
    char aligned;
    char buffer[Size];
  };
};

template<std::size_t Size>
struct AlignedCharArray<2, Size> {
  union {
    short aligned;
    char buffer[Size];
  };
};

template<std::size_t Size>
struct AlignedCharArray<4, Size> {
  union {
    int aligned;
    char buffer[Size];
  };
};

template<std::size_t Size>
struct AlignedCharArray<8, Size> {
  union {
    double aligned;
    char buffer[Size];
  };
};


// The rest of these are provided with a __declspec(align(...)) and we simply
// can't pass them by-value as function arguments on MSVC.

#define LLVM_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(x) \
  template<std::size_t Size> \
  struct AlignedCharArray<x, Size> { \
    __declspec(align(x)) char buffer[Size]; \
  };

LLVM_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(16)
LLVM_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(32)
LLVM_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(64)
LLVM_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(128)

#undef LLVM_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT

#endif // _MSC_VER

namespace detail {
template <typename T1,
          typename T2 = char, typename T3 = char, typename T4 = char,
          typename T5 = char, typename T6 = char, typename T7 = char,
          typename T8 = char, typename T9 = char, typename T10 = char>
class AlignerImpl {
  T1 t1; T2 t2; T3 t3; T4 t4; T5 t5; T6 t6; T7 t7; T8 t8; T9 t9; T10 t10;

  AlignerImpl() = delete;
};

template <typename T1,
          typename T2 = char, typename T3 = char, typename T4 = char,
          typename T5 = char, typename T6 = char, typename T7 = char,
          typename T8 = char, typename T9 = char, typename T10 = char>
union SizerImpl {
  char arr1[sizeof(T1)], arr2[sizeof(T2)], arr3[sizeof(T3)], arr4[sizeof(T4)],
       arr5[sizeof(T5)], arr6[sizeof(T6)], arr7[sizeof(T7)], arr8[sizeof(T8)],
       arr9[sizeof(T9)], arr10[sizeof(T10)];
};
} // end namespace detail

/// \brief This union template exposes a suitably aligned and sized character
/// array member which can hold elements of any of up to ten types.
///
/// These types may be arrays, structs, or any other types. The goal is to
/// expose a char array buffer member which can be used as suitable storage for
/// a placement new of any of these types. Support for more than ten types can
/// be added at the cost of more boilerplate.
template <typename T1,
          typename T2 = char, typename T3 = char, typename T4 = char,
          typename T5 = char, typename T6 = char, typename T7 = char,
          typename T8 = char, typename T9 = char, typename T10 = char>
struct AlignedCharArrayUnion : llvm::AlignedCharArray<
    alignof(llvm::detail::AlignerImpl<T1, T2, T3, T4, T5,
                                      T6, T7, T8, T9, T10>),
    sizeof(::llvm::detail::SizerImpl<T1, T2, T3, T4, T5,
                                     T6, T7, T8, T9, T10>)> {
};

/*
 * From llvm/Support/MathExtras.h
 */
static inline uint64_t NextPowerOf2(uint64_t A) {
  A |= (A >> 1);
  A |= (A >> 2);
  A |= (A >> 4);
  A |= (A >> 8);
  A |= (A >> 16);
  A |= (A >> 32);
  return A + 1;
}

/*
 * Simplified implementation given at the end of this file
 * Originally in llvm/Support/ErrorHandling.h and lib/Support/ErrorHandling.cpp
 */
void report_bad_alloc_error(const char *Reason, bool GenCrashDiag = false);

/*
 * From llvm/Support/type_traits.h
 */
/// isPodLike - This is a type trait that is used to determine whether a given
/// type can be copied around with memcpy instead of running ctors etc.
template <typename T>
struct isPodLike {
  // std::is_trivially_copyable is available in libc++ with clang, libstdc++
  // that comes with GCC 5.
#if (__has_feature(is_trivially_copyable) && defined(_LIBCPP_VERSION)) ||      \
    (defined(__GNUC__) && __GNUC__ >= 5)
  // If the compiler supports the is_trivially_copyable trait use it, as it
  // matches the definition of isPodLike closely.
  static const bool value = std::is_trivially_copyable<T>::value;
#elif __has_feature(is_trivially_copyable)
  // Use the internal name if the compiler supports is_trivially_copyable but
  // we don't know if the standard library does. This is the case for clang in
  // conjunction with libstdc++ from GCC 4.x.
  static const bool value = __is_trivially_copyable(T);
#else
  // If we don't know anything else, we can (at least) assume that all
  // non-class types are PODs.
  static const bool value = !std::is_class<T>::value;
#endif
};

// std::pair's are pod-like if their elements are.
template<typename T, typename U>
struct isPodLike<std::pair<T, U>> {
  static const bool value = isPodLike<T>::value && isPodLike<U>::value;
};

/// \brief Metafunction that determines whether the given type is either an
/// integral type or an enumeration type, including enum classes.
///
/// Note that this accepts potentially more integral types than is_integral
/// because it is based on being implicitly convertible to an integral type.
/// Also note that enum classes aren't implicitly convertible to integral
/// types, the value may therefore need to be explicitly converted before being
/// used.
template <typename T> class is_integral_or_enum {
  using UnderlyingT = typename std::remove_reference<T>::type;

public:
  static const bool value =
      !std::is_class<UnderlyingT>::value && // Filter conversion operators.
      !std::is_pointer<UnderlyingT>::value &&
      !std::is_floating_point<UnderlyingT>::value &&
      (std::is_enum<UnderlyingT>::value ||
       std::is_convertible<UnderlyingT, unsigned long long>::value);
};

/// \brief If T is a pointer, just return it. If it is not, return T&.
template<typename T, typename Enable = void>
struct add_lvalue_reference_if_not_pointer { using type = T &; };

template <typename T>
struct add_lvalue_reference_if_not_pointer<
    T, typename std::enable_if<std::is_pointer<T>::value>::type> {
  using type = T;
};

/// \brief If T is a pointer to X, return a pointer to const X. If it is not,
/// return const T.
template<typename T, typename Enable = void>
struct add_const_past_pointer { using type = const T; };

template <typename T>
struct add_const_past_pointer<
    T, typename std::enable_if<std::is_pointer<T>::value>::type> {
  using type = const typename std::remove_pointer<T>::type *;
};

template <typename T, typename Enable = void>
struct const_pointer_or_const_ref {
  using type = const T &;
};
template <typename T>
struct const_pointer_or_const_ref<
    T, typename std::enable_if<std::is_pointer<T>::value>::type> {
  using type = typename add_const_past_pointer<T>::type;
};

/*
 * From llvm/ADT/SmallVector.h
 */
/// This is all the non-templated stuff common to all SmallVectors.
class SmallVectorBase {
protected:
  void *BeginX, *EndX, *CapacityX;

protected:
  SmallVectorBase(void *FirstEl, size_t Size)
    : BeginX(FirstEl), EndX(FirstEl), CapacityX((char*)FirstEl+Size) {}

  /// This is an implementation of the grow() method which only works
  /// on POD-like data types and is out of line to reduce code duplication.
  void grow_pod(void *FirstEl, size_t MinSizeInBytes, size_t TSize);

public:
  /// This returns size()*sizeof(T).
  size_t size_in_bytes() const {
    return size_t((char*)EndX - (char*)BeginX);
  }

  /// capacity_in_bytes - This returns capacity()*sizeof(T).
  size_t capacity_in_bytes() const {
    return size_t((char*)CapacityX - (char*)BeginX);
  }

  LLVM_NODISCARD bool empty() const { return BeginX == EndX; }
};

/// This is the part of SmallVectorTemplateBase which does not depend on
/// whether the type T is a POD. The extra dummy template argument is used by
/// ArrayRef to avoid unnecessarily requiring T to be complete.
template <typename T, typename = void>
class SmallVectorTemplateCommon : public SmallVectorBase {
private:
  template <typename, unsigned> friend struct SmallVectorStorage;

  // Allocate raw space for N elements of type T.  If T has a ctor or dtor, we
  // don't want it to be automatically run, so we need to represent the space
  // as something else.  Use an array of char of sufficient alignment.
  using U = AlignedCharArrayUnion<T>;
  U FirstEl;
  // Space after 'FirstEl' is clobbered, do not add any instance vars after it.

protected:
  SmallVectorTemplateCommon(size_t Size) : SmallVectorBase(&FirstEl, Size) {}

  void grow_pod(size_t MinSizeInBytes, size_t TSize) {
    SmallVectorBase::grow_pod(&FirstEl, MinSizeInBytes, TSize);
  }

  /// Return true if this is a smallvector which has not had dynamic
  /// memory allocated for it.
  bool isSmall() const {
    return BeginX == static_cast<const void*>(&FirstEl);
  }

  /// Put this vector in a state of being small.
  void resetToSmall() {
    BeginX = EndX = CapacityX = &FirstEl;
  }

  void setEnd(T *P) { this->EndX = P; }

public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = std::reverse_iterator<iterator>;

  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

  // forward iterator creation methods.
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  iterator begin() { return (iterator)this->BeginX; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  const_iterator begin() const { return (const_iterator)this->BeginX; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  iterator end() { return (iterator)this->EndX; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  const_iterator end() const { return (const_iterator)this->EndX; }

protected:
  iterator capacity_ptr() { return (iterator)this->CapacityX; }
  const_iterator capacity_ptr() const
  { return (const_iterator)this->CapacityX; }

public:
  // reverse iterator creation methods.
  reverse_iterator rbegin()            { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const{ return const_reverse_iterator(end());}
  reverse_iterator rend()              { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const
  { return const_reverse_iterator(begin());}

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  size_type size() const { return end()-begin(); }
  size_type max_size() const { return size_type(-1) / sizeof(T); }

  /// Return the total number of elements in the currently allocated buffer.
  size_t capacity() const { return capacity_ptr() - begin(); }

  /// Return a pointer to the vector's buffer, even if empty().
  pointer data() { return pointer(begin()); }
  /// Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const { return const_pointer(begin()); }

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  reference operator[](size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  const_reference operator[](size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference front() {
    assert(!empty());
    return begin()[0];
  }
  const_reference front() const {
    assert(!empty());
    return begin()[0];
  }

  reference back() {
    assert(!empty());
    return end()[-1];
  }
  const_reference back() const {
    assert(!empty());
    return end()[-1];
  }
};

/// SmallVectorTemplateBase<isPodLike = false> - This is where we put method
/// implementations that are designed to work with non-POD-like T's.
template <typename T, bool isPodLike>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  static void destroy_range(T *S, T *E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  /// Move the range [I, E) into the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template<typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(std::make_move_iterator(I),
                            std::make_move_iterator(E), Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template<typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  /// Grow the allocated memory (without initializing new elements), doubling
  /// the size of the allocated memory. Guarantees space for at least one more
  /// element, or MinSize more elements if specified.
  void grow(size_t MinSize = 0);

public:
  void push_back(const T &Elt) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void*) this->end()) T(Elt);
    this->setEnd(this->end()+1);
  }

  void push_back(T &&Elt) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void*) this->end()) T(::std::move(Elt));
    this->setEnd(this->end()+1);
  }

  void pop_back() {
    this->setEnd(this->end()-1);
    this->end()->~T();
  }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool isPodLike>
void SmallVectorTemplateBase<T, isPodLike>::grow(size_t MinSize) {
  size_t CurCapacity = this->capacity();
  size_t CurSize = this->size();
  // Always grow, even from zero.
  size_t NewCapacity = size_t(NextPowerOf2(CurCapacity+2));
  if (NewCapacity < MinSize)
    NewCapacity = MinSize;
  T *NewElts = static_cast<T*>(malloc(NewCapacity*sizeof(T)));
  if (NewElts == nullptr)
    report_bad_alloc_error("Allocation of SmallVector element failed.");

  // Move the elements over.
  this->uninitialized_move(this->begin(), this->end(), NewElts);

  // Destroy the original elements.
  destroy_range(this->begin(), this->end());

  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!this->isSmall())
    free(this->begin());

  this->setEnd(NewElts+CurSize);
  this->BeginX = NewElts;
  this->CapacityX = this->begin()+NewCapacity;
}


/// SmallVectorTemplateBase<isPodLike = true> - This is where we put method
/// implementations that are designed to work with POD-like T's.
template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  // No need to do a destroy loop for POD's.
  static void destroy_range(T *, T *) {}

  /// Move the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template<typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    // Just do a copy.
    uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template<typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    // Arbitrary iterator types; just use the basic implementation.
    std::uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1 *I, T1 *E, T2 *Dest,
      typename std::enable_if<std::is_same<
		typename std::remove_const<T1>::type, T2>::value>::type * = nullptr) {
    // Use memcpy for PODs iterated by pointers (which includes SmallVector
    // iterators): std::uninitialized_copy optimizes to memmove, but we can
    // use memcpy here. Note that I and E are iterators and thus might be
    // invalid for memcpy if they are equal.
    if (I != E)
      memcpy(Dest, I, (E - I) * sizeof(T));
  }

  /// Double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(size_t MinSize = 0) {
    this->grow_pod(MinSize*sizeof(T), sizeof(T));
  }

public:
  void push_back(const T &Elt) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    memcpy(this->end(), &Elt, sizeof(T));
    this->setEnd(this->end()+1);
  }

  void pop_back() {
    this->setEnd(this->end()-1);
  }
};

/// This class consists of common code factored out of the SmallVector class to
/// reduce code duplication based on the SmallVector 'N' template parameter.
template <typename T>
class SmallVectorImpl : public SmallVectorTemplateBase<T, isPodLike<T>::value>{
  using SuperClass = SmallVectorTemplateBase<T, isPodLike<T>::value>;

public:
  using iterator = typename SuperClass::iterator;
  using const_iterator = typename SuperClass::const_iterator;
  using size_type = typename SuperClass::size_type;

protected:
  // Default ctor - Initialize to empty.
  explicit SmallVectorImpl(unsigned N)
    : SmallVectorTemplateBase<T, isPodLike<T>::value>(N*sizeof(T)) {
  }

public:
  SmallVectorImpl(const SmallVectorImpl &) = delete;

  ~SmallVectorImpl() {
    // Destroy the constructed elements in the vector.
    this->destroy_range(this->begin(), this->end());

    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!this->isSmall())
      free(this->begin());
  }

  void clear() {
    this->destroy_range(this->begin(), this->end());
    this->EndX = this->BeginX;
  }

  void resize(size_type N) {
    if (N < this->size()) {
      this->destroy_range(this->begin()+N, this->end());
      this->setEnd(this->begin()+N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      for (auto I = this->end(), E = this->begin() + N; I != E; ++I)
        new (&*I) T();
      this->setEnd(this->begin()+N);
    }
  }

  void resize(size_type N, const T &NV) {
    if (N < this->size()) {
      this->destroy_range(this->begin()+N, this->end());
      this->setEnd(this->begin()+N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      std::uninitialized_fill(this->end(), this->begin()+N, NV);
      this->setEnd(this->begin()+N);
    }
  }

  void reserve(size_type N) {
    if (this->capacity() < N)
      this->grow(N);
  }

  LLVM_NODISCARD T pop_back_val() {
    T Result = ::std::move(this->back());
    this->pop_back();
    return Result;
  }

  void swap(SmallVectorImpl &RHS);

  /// Add the specified range to the end of the SmallVector.
  template <typename in_iter,
            typename = typename std::enable_if<std::is_convertible<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>::value>::type>
  void append(in_iter in_start, in_iter in_end) {
    size_type NumInputs = std::distance(in_start, in_end);
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(this->size()+NumInputs);

    // Copy the new elements over.
    this->uninitialized_copy(in_start, in_end, this->end());
    this->setEnd(this->end() + NumInputs);
  }

  /// Add the specified range to the end of the SmallVector.
  void append(size_type NumInputs, const T &Elt) {
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(this->size()+NumInputs);

    // Copy the new elements over.
    std::uninitialized_fill_n(this->end(), NumInputs, Elt);
    this->setEnd(this->end() + NumInputs);
  }

  void append(std::initializer_list<T> IL) {
    append(IL.begin(), IL.end());
  }

  // FIXME: Consider assigning over existing elements, rather than clearing &
  // re-initializing them - for all assign(...) variants.

  void assign(size_type NumElts, const T &Elt) {
    clear();
    if (this->capacity() < NumElts)
      this->grow(NumElts);
    this->setEnd(this->begin()+NumElts);
    std::uninitialized_fill(this->begin(), this->end(), Elt);
  }

  template <typename in_iter,
            typename = typename std::enable_if<std::is_convertible<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>::value>::type>
  void assign(in_iter in_start, in_iter in_end) {
    clear();
    append(in_start, in_end);
  }

  void assign(std::initializer_list<T> IL) {
    clear();
    append(IL);
  }

  iterator erase(const_iterator CI) {
    // Just cast away constness because this is a non-const member function.
    iterator I = const_cast<iterator>(CI);

    assert(I >= this->begin() && "Iterator to erase is out of bounds.");
    assert(I < this->end() && "Erasing at past-the-end iterator.");

    iterator N = I;
    // Shift all elts down one.
    std::move(I+1, this->end(), I);
    // Drop the last elt.
    this->pop_back();
    return(N);
  }

  iterator erase(const_iterator CS, const_iterator CE) {
    // Just cast away constness because this is a non-const member function.
    iterator S = const_cast<iterator>(CS);
    iterator E = const_cast<iterator>(CE);

    assert(S >= this->begin() && "Range to erase is out of bounds.");
    assert(S <= E && "Trying to erase invalid range.");
    assert(E <= this->end() && "Trying to erase past the end.");

    iterator N = S;
    // Shift all elts down.
    iterator I = std::move(E, this->end(), S);
    // Drop the last elts.
    this->destroy_range(I, this->end());
    this->setEnd(I);
    return(N);
  }

  iterator insert(iterator I, T &&Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      this->push_back(::std::move(Elt));
      return this->end()-1;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = I-this->begin();
      this->grow();
      I = this->begin()+EltNo;
    }

    ::new ((void*) this->end()) T(::std::move(this->back()));
    // Push everything else over.
    std::move_backward(I, this->end()-1, this->end());
    this->setEnd(this->end()+1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    T *EltPtr = &Elt;
    if (I <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *I = ::std::move(*EltPtr);
    return I;
  }

  iterator insert(iterator I, const T &Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      this->push_back(Elt);
      return this->end()-1;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = I-this->begin();
      this->grow();
      I = this->begin()+EltNo;
    }
    ::new ((void*) this->end()) T(std::move(this->back()));
    // Push everything else over.
    std::move_backward(I, this->end()-1, this->end());
    this->setEnd(this->end()+1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    const T *EltPtr = &Elt;
    if (I <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *I = *EltPtr;
    return I;
  }

  iterator insert(iterator I, size_type NumToInsert, const T &Elt) {
    // Convert iterator to elt# to avoid invalidating iterator when we
	// reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(NumToInsert, Elt);
      return this->begin()+InsertElt;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of
	// the range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(I, OldEnd-NumToInsert, OldEnd);

      std::fill_n(I, NumToInsert, Elt);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_move(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    std::fill_n(I, NumOverwritten, Elt);

    // Insert the non-overwritten middle part.
    std::uninitialized_fill_n(OldEnd, NumToInsert-NumOverwritten, Elt);
    return I;
  }

  template <typename ItTy,
            typename = typename std::enable_if<std::is_convertible<
                typename std::iterator_traits<ItTy>::iterator_category,
                std::input_iterator_tag>::value>::type>
  iterator insert(iterator I, ItTy From, ItTy To) {
    // Convert iterator to elt# to avoid invalidating iterator when we
	// reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(From, To);
      return this->begin()+InsertElt;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    size_t NumToInsert = std::distance(From, To);

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of
	// the range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(I, OldEnd-NumToInsert, OldEnd);

      std::copy(From, To, I);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_move(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    for (T *J = I; NumOverwritten > 0; --NumOverwritten) {
      *J = *From;
      ++J; ++From;
    }

    // Insert the non-overwritten middle part.
    this->uninitialized_copy(From, To, OldEnd);
    return I;
  }

  void insert(iterator I, std::initializer_list<T> IL) {
    insert(I, IL.begin(), IL.end());
  }

  template <typename... ArgTypes> void emplace_back(ArgTypes &&... Args) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void *)this->end()) T(std::forward<ArgTypes>(Args)...);
    this->setEnd(this->end() + 1);
  }

  SmallVectorImpl &operator=(const SmallVectorImpl &RHS);

  SmallVectorImpl &operator=(SmallVectorImpl &&RHS);

  bool operator==(const SmallVectorImpl &RHS) const {
    if (this->size() != RHS.size()) return false;
    return std::equal(this->begin(), this->end(), RHS.begin());
  }
  bool operator!=(const SmallVectorImpl &RHS) const {
    return !(*this == RHS);
  }

  bool operator<(const SmallVectorImpl &RHS) const {
    return std::lexicographical_compare(this->begin(), this->end(),
                                        RHS.begin(), RHS.end());
  }

  /// Set the array size to \p N, which the current array must have enough
  /// capacity for.
  ///
  /// This does not construct or destroy any elements in the vector.
  ///
  /// Clients can use this in conjunction with capacity() to write past the end
  /// of the buffer when they know that more elements are available, and only
  /// update the size later. This avoids the cost of value initializing
  /// elements which will only be overwritten.
  void set_size(size_type N) {
    assert(N <= this->capacity());
    this->setEnd(this->begin() + N);
  }
};

template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T> &RHS) {
  if (this == &RHS) return;

  // We can only avoid copying elements if neither vector is small.
  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->BeginX, RHS.BeginX);
    std::swap(this->EndX, RHS.EndX);
    std::swap(this->CapacityX, RHS.CapacityX);
    return;
  }
  if (RHS.size() > this->capacity())
    this->grow(RHS.size());
  if (this->size() > RHS.capacity())
    RHS.grow(this->size());

  // Swap the shared elements.
  size_t NumShared = this->size();
  if (NumShared > RHS.size()) NumShared = RHS.size();
  for (size_type i = 0; i != NumShared; ++i)
    std::swap((*this)[i], RHS[i]);

  // Copy over the extra elts.
  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    this->uninitialized_copy(this->begin()+NumShared, this->end(), RHS.end());
    RHS.setEnd(RHS.end()+EltDiff);
    this->destroy_range(this->begin()+NumShared, this->end());
    this->setEnd(this->begin()+NumShared);
  } else if (RHS.size() > this->size()) {
    size_t EltDiff = RHS.size() - this->size();
    this->uninitialized_copy(RHS.begin()+NumShared, RHS.end(), this->end());
    this->setEnd(this->end() + EltDiff);
    this->destroy_range(RHS.begin()+NumShared, RHS.end());
    RHS.setEnd(RHS.begin()+NumShared);
  }
}

template <typename T>
SmallVectorImpl<T> &SmallVectorImpl<T>::
  operator=(const SmallVectorImpl<T> &RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd;
    if (RHSSize)
      NewEnd = std::copy(RHS.begin(), RHS.begin()+RHSSize, this->begin());
    else
      NewEnd = this->begin();

    // Destroy excess elements.
    this->destroy_range(NewEnd, this->end());

    // Trim.
    this->setEnd(NewEnd);
    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: don't do this if they're efficiently moveable.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::copy(RHS.begin(), RHS.begin()+CurSize, this->begin());
  }

  // Copy construct the new elements in place.
  this->uninitialized_copy(RHS.begin()+CurSize, RHS.end(),
                           this->begin()+CurSize);

  // Set end.
  this->setEnd(this->begin()+RHSSize);
  return *this;
}

template <typename T>
SmallVectorImpl<T> &SmallVectorImpl<T>::operator=(SmallVectorImpl<T> &&RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If the RHS isn't small, clear this vector and then steal its buffer.
  if (!RHS.isSmall()) {
    this->destroy_range(this->begin(), this->end());
    if (!this->isSmall()) free(this->begin());
    this->BeginX = RHS.BeginX;
    this->EndX = RHS.EndX;
    this->CapacityX = RHS.CapacityX;
    RHS.resetToSmall();
    return *this;
  }

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd = this->begin();
    if (RHSSize)
      NewEnd = std::move(RHS.begin(), RHS.end(), NewEnd);

    // Destroy excess elements and trim the bounds.
    this->destroy_range(NewEnd, this->end());
    this->setEnd(NewEnd);

    // Clear the RHS.
    RHS.clear();

    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: this may not actually make any sense if we can efficiently move
  // elements.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::move(RHS.begin(), RHS.begin()+CurSize, this->begin());
  }

  // Move-construct the new elements in place.
  this->uninitialized_move(RHS.begin()+CurSize, RHS.end(),
                           this->begin()+CurSize);

  // Set end.
  this->setEnd(this->begin()+RHSSize);

  RHS.clear();
  return *this;
}

/// Storage for the SmallVector elements which aren't contained in
/// SmallVectorTemplateCommon. There are 'N-1' elements here. The remaining '1'
/// element is in the base class. This is specialized for the N=1 and N=0 cases
/// to avoid allocating unnecessary storage.
template <typename T, unsigned N>
struct SmallVectorStorage {
  typename SmallVectorTemplateCommon<T>::U InlineElts[N - 1];
};
template <typename T> struct SmallVectorStorage<T, 1> {};
template <typename T> struct SmallVectorStorage<T, 0> {};

/// This is a 'vector' (really, a variable-sized array), optimized
/// for the case when the array is small.  It contains some number of elements
/// in-place, which allows it to avoid heap allocation when the actual number
/// of elements is below that threshold.  This allows normal "small" cases to
/// be fast without losing generality for large inputs.
///
/// Note that this does not attempt to be exception safe.
///
template <typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
  /// Inline space for elements which aren't stored in the base class.
  SmallVectorStorage<T, N> Storage;

public:
  SmallVector() : SmallVectorImpl<T>(N) {}

  explicit SmallVector(size_t Size, const T &Value = T())
    : SmallVectorImpl<T>(N) {
    this->assign(Size, Value);
  }

  template <typename ItTy,
            typename = typename std::enable_if<std::is_convertible<
                typename std::iterator_traits<ItTy>::iterator_category,
                std::input_iterator_tag>::value>::type>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
    this->append(S, E);
  }

  template <typename RangeTy>
  explicit SmallVector(const iterator_range<RangeTy> &R)
      : SmallVectorImpl<T>(N) {
    this->append(R.begin(), R.end());
  }

  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(N) {
    this->assign(IL);
  }

  SmallVector(const SmallVector &RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(RHS);
  }

  const SmallVector &operator=(const SmallVector &RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  SmallVector(SmallVector &&RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  SmallVector(SmallVectorImpl<T> &&RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  const SmallVector &operator=(SmallVector &&RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  const SmallVector &operator=(SmallVectorImpl<T> &&RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  const SmallVector &operator=(std::initializer_list<T> IL) {
    this->assign(IL);
    return *this;
  }
};

template<typename T, unsigned N>
static inline size_t capacity_in_bytes(const SmallVector<T, N> &X) {
  return X.capacity_in_bytes();
}

} // end namespace llvm

namespace std {

  /// Implement std::swap in terms of SmallVector swap.
  template<typename T>
  inline void
  swap(llvm::SmallVectorImpl<T> &LHS, llvm::SmallVectorImpl<T> &RHS) {
    LHS.swap(RHS);
  }

  /// Implement std::swap in terms of SmallVector swap.
  template<typename T, unsigned N>
  inline void
  swap(llvm::SmallVector<T, N> &LHS, llvm::SmallVector<T, N> &RHS) {
    LHS.swap(RHS);
  }

} // end namespace std

// If the compiler supports detecting whether a class is final, define
// an LLVM_IS_FINAL macro. If it cannot be defined properly, this
// macro will be left undefined.
#if __cplusplus >= 201402L
#define LLVM_IS_FINAL(Ty) std::is_final<Ty>()
#elif __has_feature(is_final) || LLVM_GNUC_PREREQ(4, 7, 0)
#define LLVM_IS_FINAL(Ty) __is_final(Ty)
#endif

#ifdef LLVM_DEFINED_HAS_FEATURE
#undef __has_feature
#endif

#ifdef LLVM_SMALL_VECTOR_IMPLEMENTATION

namespace llvm {

/*
 * From lib/Support/SmallVector.cpp
 */
/// grow_pod - This is an implementation of the grow() method which only works
/// on POD-like datatypes and is out of line to reduce code duplication.
void SmallVectorBase::grow_pod(void *FirstEl, size_t MinSizeInBytes,
                               size_t TSize) {
  size_t CurSizeBytes = size_in_bytes();
  size_t NewCapacityInBytes = 2 * capacity_in_bytes() + TSize; // Always grow.
  if (NewCapacityInBytes < MinSizeInBytes)
    NewCapacityInBytes = MinSizeInBytes;

  void *NewElts;
  if (BeginX == FirstEl) {
    NewElts = malloc(NewCapacityInBytes);
    if (NewElts == nullptr)
      report_bad_alloc_error("Allocation of SmallVector element failed.");

    // Copy the elements over.  No need to run dtors on PODs.
    memcpy(NewElts, this->BeginX, CurSizeBytes);
  } else {
    // If this wasn't grown from the inline copy, grow the allocated space.
    NewElts = realloc(this->BeginX, NewCapacityInBytes);
    if (NewElts == nullptr)
      report_bad_alloc_error("Reallocation of SmallVector element failed.");
  }

  this->EndX = (char*)NewElts+CurSizeBytes;
  this->BeginX = NewElts;
  this->CapacityX = (char*)this->BeginX + NewCapacityInBytes;
}

/*
 * Simplified implementation
 * Originally in llvm/Support/ErrorHandling.h and lib/Support/ErrorHandling.cpp
 */
void report_bad_alloc_error(const char *Reason, bool GenCrashDiag) {
#ifdef LLVM_SMALL_VECTOR_BAD_ALLOC
	LLVM_SMALL_VECTOR_BAD_ALLOC(Reason);
#else
	fputs("LLVM SmallVector: Out of memory\n", stderr);
	abort();
#endif
}

} // namespace llvm

#endif // LLVM_SMALL_VECTOR_IMPLEMENTATION

#endif // LLVM_ADT_SMALLVECTOR_H
