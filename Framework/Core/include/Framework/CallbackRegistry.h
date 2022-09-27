// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CALLBACKREGISTRY_H
#define FRAMEWORK_CALLBACKREGISTRY_H

/// @file   CallbackRegistry.h
/// @author Matthias Richter
/// @since  2018-04-26
/// @brief  A generic registry for callbacks

#include "Framework/TypeTraits.h"
#include "Framework/RuntimeError.h"
#include <tuple>
#include <functional>

namespace o2::framework
{

template <typename KeyT, KeyT _id, typename CallbackT>
struct RegistryPair {
  using id = std::integral_constant<KeyT, _id>;
  using type = CallbackT;
  type callback;
};

template <typename CallbackId, typename... Args>
class CallbackRegistry
{
 public:
  // FIXME:
  // - add more checks
  //   - recursive check of the argument pack
  //   - required to be of type RegistryPair
  //   - callback type is specialization of std::function
  // - extend to variable return type

  static constexpr std::size_t size = sizeof...(Args);
  using CallbackStore = std::tuple<Args...>;

  // set callback for slot
  template <typename U>
  void set(CallbackId id, U&& cb)
  {
    set<size>(id, cb);
  }

  // execute callback at specified slot with argument pack
  template <typename... TArgs>
  auto operator()(CallbackId id, TArgs&&... args)
  {
    exec<size>(id, std::forward<TArgs>(args)...);
  }

 private:
  // helper trait to check whether class has a constructor taking callback as argument
  template <class T, typename CB>
  struct has_matching_callback {
    template <class U, typename V>
    static int check(decltype(U(std::declval<V>()))*);

    template <typename U, typename V>
    static char check(...);

    static const bool value = sizeof(check<T, CB>(nullptr)) == sizeof(int);
  };

  // set the callback function of the specified id
  // this iterates over defined slots and sets the matching callback
  template <std::size_t pos, typename U>
  void set(CallbackId id, U&& cb)
  {
    if constexpr (pos != 0) {
      if (std::tuple_element<pos - 1, CallbackStore>::type::id::value != id) {
        return set<pos - 1>(id, cb);
      }
      // note: there are two substitutions, the one for callback matching the slot type sets the
      // callback, while the other substitution should never be called
      setAt<pos - 1, typename std::tuple_element<pos - 1, CallbackStore>::type::type>(cb);
    }
  }

  // set the callback at specified slot position
  template <std::size_t pos, typename U, typename F>
  void setAt(F&& cb)
  {
    if constexpr (has_matching_callback<U, F>::value == true) {
      // create a new std::function object and init with the callback function
      std::get<pos>(mStore).callback = (U)(cb);
    } else {
      throw runtime_error_f("mismatch in function substitution at position %d", pos);
    }
  }

  // exec callback of specified id
  template <std::size_t pos, typename... TArgs>
  void exec(CallbackId id, TArgs&&... args)
  {
    if constexpr (pos != 0) {
      if (std::tuple_element<pos - 1, CallbackStore>::type::id::value != id) {
        return exec<pos - 1>(id, std::forward<TArgs>(args)...);
      }
      // build a callable function type from the result type of the slot and the
      // argument pack, this is used to selcet the matching substitution
      using FunT = typename std::tuple_element<pos - 1, CallbackStore>::type::type;
      using ResT = typename FunT::result_type;
      using CheckT = std::function<ResT(TArgs...)>;
      execAt<pos - 1, FunT, CheckT>(std::forward<TArgs>(args)...);
    }
  }

  // exec callback at specified slot
  template <std::size_t pos, typename U, typename V, typename... TArgs>
  void execAt(TArgs&&... args)
  {
    if constexpr (std::is_same_v<U, V> == true) {
      if (std::get<pos>(mStore).callback) {
        std::get<pos>(mStore).callback(std::forward<TArgs>(args)...);
      }
    }
  }

  // store of RegistryPairs
  std::tuple<Args...> mStore;
};
} // namespace o2::framework
#endif // FRAMEWORK_CALLBACKREGISTRY_H
