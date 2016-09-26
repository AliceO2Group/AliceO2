//-*- Mode: C++ -*-

#ifndef CODINGMODELDISPATCHER_H
#define CODINGMODELDISPATCHER_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   CodingModelDispatcher.h
//  @author Matthias Richter
//  @since  2016-09-11
//  @brief  Runtime dispatcher interface for probability model definitions

#include "mpl_tools.h"
#include "runtime_container.h"

namespace ALICE {
namespace O2 {

/**
 * @class CodingModelDispatcher Runtime dispatcher interface
 * @brief Runtime dispatcher interface for probability model definitions
 *
 * ModelDefinition single coding model or mpl sequence of models
 */
template<typename ModelDefinition>
class CodingModelDispatcher {
public:
  CodingModelDispatcher() : mPosition(0), mContainer() {}
  ~CodingModelDispatcher() {}

  typedef CodingModelDispatcher<ModelDefinition> self_type;

  // make_mpl_vector traits makes sure that an mpl sequence is used further on
  // if the original type is not a sequence it is wrapped into an mpl vector with
  // the original type as the only element
  typedef typename mpl::make_mpl_vector<ModelDefinition>::type definition_type;

  // the runtime container type is the heart of the dispatcher to runtime objects
  // of the sequence of data types which define the probability model
  typedef typename create_rtc<definition_type, RuntimeContainer<>>::type container_type;

  /// get the number of models in the definition
  static int getNumberOfModels() {return boost::mpl::size<definition_type>::value;}

  /// return highest stage of runtime container
  container_type& operator*() {return mContainer;}

  /// functor to execute encoding on runtime container level
  template<typename CodeType, typename ValueType>
  class encode {
  public:
    encode(ValueType _v, CodeType& _code, uint16_t& _codeLength)
      : code(_code), value(_v), codeLength(_codeLength) {}
    ~encode() {}

    typedef bool return_type;

    template<typename T>
    return_type operator()(T& stage) {
      (*stage).Encode(value, code, codeLength);
      return true;
    }

  private:
    CodeType& code;
    ValueType value;
    uint16_t& codeLength;
  };

  /**
   * Encode a value
   */
  template<typename CodeType, typename ValueType>
  bool Encode(ValueType v, CodeType& code, uint16_t& codeLength) {
    bool result = mContainer.apply(mPosition, encode<CodeType, ValueType>(v, code, codeLength));
    if (++mPosition >= getNumberOfModels()) mPosition = 0;
    return result;
  }

  /// Functor to execute decoding on runtime container level
  template<typename CodeType, typename ValueType>
  class decode {
  public:
    decode(ValueType& _v, CodeType _code, uint16_t& _codeLength)
      : code(_code), value(_v), codeLength(_codeLength) {}
    ~decode() {}

    typedef bool return_type;

    template<typename T>
    return_type operator()(T& stage) {
      (*stage).Decode(value, code, codeLength);
      return true;
    }

  private:
    CodeType code;
    ValueType& value;
    uint16_t& codeLength;
  };

  /**
   * Decode a code sequence
   * Code direction is controlled by template parameter orderMSB and can be
   * either from MSB to LSB or LSB to MSB.
   */
  template<typename ValueType, typename CodeType, bool orderMSB = true>
  bool Decode(ValueType& v, CodeType code, uint16_t& codeLength) {
    bool result = mContainer.apply(mPosition, decode<CodeType, ValueType>(v, code, codeLength));
    if (++mPosition >= getNumberOfModels()) mPosition = 0;
    return result;
  }

private:
  /// position for cyclic dispatch
  int mPosition;
  /// the runtime container
  container_type mContainer;
};

}; // namespace O2
}; // namespace ALICE

#endif
