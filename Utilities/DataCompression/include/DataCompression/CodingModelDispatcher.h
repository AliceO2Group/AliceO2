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

/* Local Variables:  */
/* mode: c++         */
/* End:              */

#ifndef CODINGMODELDISPATCHER_H
#define CODINGMODELDISPATCHER_H

/// @file   CodingModelDispatcher.h
/// @author Matthias Richter
/// @since  2016-09-11
/// @brief  Runtime dispatcher interface for probability model definitions

#include "mpl_tools.h"
#include "runtime_container.h"
#include <iostream>
#include <fstream>
#include <boost/type.hpp>
#include <boost/mpl/for_each.hpp>

using namespace gNeric;

namespace o2
{
namespace data_compression
{

/**
 * @class CodingModelDispatcher Runtime dispatcher interface
 * @brief Runtime dispatcher interface for probability model definitions
 *
 * ModelDefinition single coding model or mpl sequence of models
 *
 * TODO:
 * - consistency check for coding direction, all model definitions have to obey
 *   same direction
 * - probably one should also require the same code type for all definitions, at
 *   least in the codec. Multiple code types do not make much sense in the codec
 */
template <typename ModelDefinition>
class CodingModelDispatcher
{
 public:
  CodingModelDispatcher() : mPosition(0), mContainer() {}
  ~CodingModelDispatcher() = default;

  using self_type = CodingModelDispatcher<ModelDefinition>;

  // make_mpl_vector traits makes sure that an mpl sequence is used further on
  // if the original type is not a sequence it is wrapped into an mpl vector with
  // the original type as the only element
  using definition_type = typename mpl::make_mpl_vector<ModelDefinition>::type;

  // the runtime container type is the heart of the dispatcher to runtime objects
  // of the sequence of data types which define the probability model
  using container_type = typename create_rtc<definition_type, RuntimeContainer<>>::type;

  using code_type = typename container_type::wrapped_type::code_type;

  /// get the number of models in the definition
  static int getNumberOfModels() { return boost::mpl::size<definition_type>::value; }

  /// return highest stage of runtime container
  container_type& operator*() { return mContainer; }

  /// functor to add weight to probability model at runtime container level
  template <typename ValueType, typename WeightType>
  class addWeightFctr
  {
   public:
    addWeightFctr(ValueType _v, WeightType _w) : value(_v), weight(_w) {}
    ~addWeightFctr() {}

    using return_type = bool;

    template <typename T>
    return_type operator()(T& stage)
    {
      // the addWeight function belongs to the probability model as base
      // of the specific model; funcions of the base can be accessed by
      // static casting. This avoids an extra level of function calls.
      return static_cast<typename T::wrapped_type::base_type&>(*stage).addWeight(value, weight);
    }

   private:
    ValueType value;
    WeightType weight;
  };

  /**
   * add weight to current model
   *
   * Dispatcher increments to the next model definition after decoding if
   * parameter switchToNextModel is true.
   */
  template <typename ValueType, typename WeightType>
  bool addWeight(ValueType v, WeightType w, bool switchToNextModel = true)
  {
    bool result = mContainer.apply(mPosition, addWeightFctr<ValueType, WeightType>(v, w));
    if (switchToNextModel && ++mPosition >= getNumberOfModels()) {
      mPosition = 0;
    }
    return result;
  }

  /**
   * init model
   */
  class initFctr
  {
   public:
    initFctr(container_type& container) : mContainer(container) {}
    ~initFctr() {}

    using return_type = int;

    template <typename T>
    return_type operator()(boost::type<T>)
    {
      T& stage = static_cast<T&>(mContainer);
      return (*stage).init();
    }

   private:
    container_type& mContainer;
  };

  /**
   * init dispatcher and models
   */
  int init()
  {
    mPosition = 0;
    boost::mpl::for_each<typename container_type::types, boost::type<boost::mpl::_>>(initFctr(mContainer));
    return 0;
  }

  /**
   * TODO: this is tailored to HuffmanCodec for the moment, some generic interface
   * has to come
   */
  class generateFctr
  {
   public:
    generateFctr(container_type& container) : mContainer(container) {}
    ~generateFctr() {}

    using return_type = int;

    template <typename T>
    return_type operator()(boost::type<T>)
    {
      T& stage = static_cast<T&>(mContainer);
      return (*stage).GenerateHuffmanTree();
    }

   private:
    container_type& mContainer;
  };

  /**
   * TODO: maybe 'generate' is not the appropriate name
   */
  int generate()
  {
    boost::mpl::for_each<typename container_type::types, boost::type<boost::mpl::_>>(generateFctr(mContainer));
    return 0;
  }

  /// functor to execute encoding on runtime container level
  template <typename CodeType, typename ValueType>
  class encodeFctr
  {
   public:
    encodeFctr(ValueType _v, CodeType& _code, uint16_t& _codeLength) : code(_code), value(_v), codeLength(_codeLength)
    {
    }
    ~encodeFctr() {}

    using return_type = bool;

    template <typename T>
    return_type operator()(T& stage)
    {
      code = (*stage).Encode(value, codeLength);
      return true;
    }

   private:
    CodeType& code;
    ValueType value;
    uint16_t& codeLength;
  };

  /**
   * Encode a value
   *
   * Dispatcher increments to the next model definition after decoding if
   * parameter switchToNextModel is true.
   */
  template <typename CodeType, typename ValueType>
  bool encode(ValueType v, CodeType& code, uint16_t& codeLength, bool switchToNextModel = true)
  {
    bool result = mContainer.apply(mPosition, encodeFctr<CodeType, ValueType>(v, code, codeLength));
    if (switchToNextModel && ++mPosition >= getNumberOfModels()) {
      mPosition = 0;
    }
    return result;
  }

  /// Functor to execute decoding on runtime container level
  template <typename CodeType, typename ValueType>
  class decodeFctr
  {
   public:
    decodeFctr(ValueType& _v, CodeType _code, uint16_t& _codeLength) : code(_code), value(_v), codeLength(_codeLength)
    {
    }
    ~decodeFctr() {}

    using return_type = bool;

    template <typename T>
    return_type operator()(T& stage)
    {
      value = (*stage).Decode(code, codeLength);
      return true;
    }

   private:
    CodeType code;
    ValueType& value;
    uint16_t& codeLength;
  };

  /**
   * Decode a code sequence
   * Code direction can be either from MSB to LSB or LSB to MSB, controlled
   * by template parameter orderMSB of the probability model.
   *
   * Dispatcher increments to the next model definition after decoding if
   * parameter switchToNextModel is true.
   */
  template <typename ValueType, typename CodeType>
  bool decode(ValueType& v, CodeType code, uint16_t& codeLength, bool switchToNextModel = true)
  {
    bool result = mContainer.apply(mPosition, decodeFctr<CodeType, ValueType>(v, code, codeLength));
    if (switchToNextModel && ++mPosition >= getNumberOfModels()) {
      mPosition = 0;
    }
    return result;
  }

  class getCodingDirectionFctr
  {
   public:
    using return_type = bool;
    template <typename T>
    return_type operator()(T& stage)
    {
      return T::wrapped_type::orderMSB;
    }
  };

  /**
   * Get coding direction for model at current position
   */
  bool getCodingDirection() { return mContainer.apply(mPosition, getCodingDirectionFctr()); }

  /// write functor
  class writeFctr
  {
   public:
    writeFctr(std::ostream& out, container_type& container) : mOut(out), mContainer(container) {}
    ~writeFctr() {}

    using return_type = std::ostream&;

    template <typename T>
    return_type operator()(boost::type<T>)
    {
      T& stage = static_cast<T&>(mContainer);
      if (T::level::value > 0) {
        mOut << std::endl; // blank line between dumps
      }
      mOut << T::level::value << " " << (*stage).getName() << std::endl;
      (*stage).write(mOut);
      return mOut;
    }

   private:
    std::ostream& mOut;
    container_type& mContainer;
  };

  /**
   * Write configuration
   *
   * TODO: introduce a general storage policy, a text file is used for now
   */
  int write(const char* filename = nullptr)
  {
    std::ofstream ofile(filename);
    boost::mpl::for_each<typename container_type::types, boost::type<boost::mpl::_>>(
      writeFctr(ofile.good() ? ofile : std::cout, mContainer));
    ofile.close();
    return 0;
  }

  /// read functor
  class readFctr
  {
   public:
    readFctr(std::istream& in, container_type& container) : mIn(in), mContainer(container) {}
    ~readFctr() {}

    using return_type = bool;

    template <typename T>
    return_type operator()(boost::type<T>)
    {
      T& stage = static_cast<T&>(mContainer);
      std::string level, name, remaining;
      mIn >> level;
      mIn >> name;
      if (!mIn) {
        return false;
      }
      if (std::stoi(level) != T::level::value || name.compare((*stage).getName())) {
        std::cerr << "Format error: expecting level '" << T::level::value << "' and name '" << (*stage).getName()
                  << "', got: " << level << " " << name << std::endl;
      }
      std::cout << "reading configuration for model " << name << std::endl;
      std::getline(mIn, remaining); // flush the current line
      (*stage).read(mIn);
      return true;
    }

   private:
    std::istream& mIn;
    container_type& mContainer;
  };

  /**
   * Read configuration
   *
   * TODO: introduce a general storage policy, a text file is used for now
   */
  int read(const char* filename)
  {
    std::ifstream input(filename);
    if (!input.good()) {
      return -1;
    }
    // TODO: probably need mpl fold here to propagate the return value
    boost::mpl::for_each<typename container_type::types, boost::type<boost::mpl::_>>(readFctr(input, mContainer));
    return 0;
  }

 private:
  /// position for cyclic dispatch
  int mPosition;
  /// the runtime container
  container_type mContainer;
};

} // namespace data_compression
} // namespace o2

#endif
