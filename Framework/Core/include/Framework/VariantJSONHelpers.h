// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_VARIANTJSONHELPERS_H
#define FRAMEWORK_VARIANTJSONHELPERS_H

#include "Framework/Variant.h"

#include <rapidjson/reader.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/error/en.h>

#include <stack>
#include <iostream>
#include <sstream>

namespace o2::framework
{
namespace
{
template <VariantType V>
struct VariantReader : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, VariantReader<V>> {
  using Ch = rapidjson::UTF8<>::Ch;
  using SizeType = rapidjson::SizeType;

  enum struct State {
    IN_START,
    IN_STOP,
    IN_DATA,
    IN_KEY,
    IN_ARRAY,
    IN_ROW,
    IN_ERROR
  };

  VariantReader()
    : states{}
  {
    debug << "Start" << std::endl;
    states.push(State::IN_START);
  }

  bool Null()
  {
    debug << "Null value encountered" << std::endl;
    return true;
  }

  bool Int(int i)
  {
    debug << "Int(" << i << ")" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if constexpr (!(V == VariantType::ArrayInt || V == VariantType::Array2DInt)) {
      debug << "Integer value in non-integer variant" << std::endl;
      states.push(State::IN_ERROR);
      return true;
    } else {
      if (states.top() == State::IN_ARRAY || states.top() == State::IN_ROW) {
        debug << "added to array" << std::endl;
        accumulatedData.push_back(i);
        return true;
      }
    }
    states.push(State::IN_ERROR);
    return true;
  }

  bool Uint(unsigned i)
  {
    debug << "Uint -> Int" << std::endl;
    return Int(static_cast<int>(i));
  }

  bool Int64(int64_t i)
  {
    debug << "Int64 -> Int" << std::endl;
    return Int(static_cast<int>(i));
  }

  bool Uint64(uint64_t i)
  {
    debug << "Uint64 -> Int" << std::endl;
    return Int(static_cast<int>(i));
  }

  bool Double(double d)
  {
    debug << "Double(" << d << ")" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if constexpr (!(V == VariantType::ArrayDouble || V == VariantType::Array2DDouble || V == VariantType::ArrayFloat || V == VariantType::Array2DFloat)) {
      states.push(State::IN_ERROR);
      return true;
    }
    if (states.top() == State::IN_ARRAY || states.top() == State::IN_ROW) {
      if constexpr (V == VariantType::ArrayDouble || V == VariantType::Array2DDouble) {
        debug << "added to array as double" << std::endl;
        accumulatedData.push_back(d);
        return true;
      } else if constexpr (V == VariantType::ArrayFloat || V == VariantType::Array2DFloat) {
        debug << "added to array as float" << std::endl;
        accumulatedData.push_back(static_cast<float>(d));
        return true;
      }
    }
    states.push(State::IN_ERROR);
    return true;
  }

  bool Bool(bool b)
  {
    debug << "Bool(" << b << ")" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if constexpr (V != VariantType::ArrayBool) {
      states.push(State::IN_ERROR);
      return false;
    } else {
      if (states.top() == State::IN_ARRAY || states.top() == State::IN_ROW) {
        debug << "added to array" << std::endl;
        accumulatedData.push_back(b);
        return true;
      }
    }
  }

  bool String(const Ch* str, SizeType, bool)
  {
    debug << "String(" << str << ")" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if constexpr (V != VariantType::ArrayString) {
      states.push(State::IN_ERROR);
      return true;
    } else {
      if (states.top() == State::IN_ARRAY || states.top() == State::IN_ROW) {
        debug << "added to array" << std::endl;
        accumulatedData.push_back(str);
        return true;
      }
      states.push(State::IN_ERROR);
      return true;
    }
  }

  bool StartObject()
  {
    debug << "StartObject()" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if (states.top() == State::IN_START) {
      states.push(State::IN_DATA);
      return true;
    }
    states.push(State::IN_ERROR);
    return true;
  }

  bool Key(const Ch* str, SizeType, bool)
  {
    debug << "Key(" << str << ")" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if (states.top() == State::IN_DATA || states.top() == State::IN_KEY) {
      states.push(State::IN_KEY);
      currentKey = str;
      return true;
    }
    states.push(State::IN_ERROR);
    return true;
  }

  bool EndObject(SizeType)
  {
    debug << "EndObject()" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if (states.top() == State::IN_KEY) {
      // finish up key
      if constexpr (isArray<V>()) {
        debug << "creating 1d-array variant" << std::endl;
        result = Variant(accumulatedData);
      } else if constexpr (isArray2D<V>()) {
        debug << "creating 2d-array variant" << std::endl;
        result = Variant(Array2D{accumulatedData, rows, cols});
      }
      states.push(State::IN_STOP);
      return true;
    }
    states.push(State::IN_ERROR);
    return true;
  }

  bool StartArray()
  {
    debug << "StartArray()" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if (states.top() == State::IN_KEY) {
      states.push(State::IN_ARRAY);
      rows = 1;
      return true;
    } else if (states.top() == State::IN_ARRAY) {
      if constexpr (isArray2D<V>()) {
        states.push(State::IN_ROW);
        ++rows;
        return true;
      }
    }
    states.push(State::IN_ERROR);
    return true;
  }

  bool EndArray(SizeType elementCount)
  {
    debug << "EndArray()" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if (states.top() == State::IN_ARRAY) {
      //finish up array
      states.pop();
      if constexpr (isArray<V>()) {
        cols = elementCount;
      } else if constexpr (isArray2D<V>()) {
        rows = elementCount;
      }
      return true;
    } else if (states.top() == State::IN_ROW) {
      if constexpr (isArray2D<V>()) {
        cols = elementCount;
      }
      //finish up row
      states.pop();
      return true;
    }
    states.push(State::IN_ERROR);
    return true;
  }

  std::stack<State> states;
  std::ostringstream debug;

  uint32_t rows;
  uint32_t cols;
  std::string currentKey;
  std::vector<typename variant_array_element_type<V>::type> accumulatedData;
  Variant result;
};

template <VariantType V>
void writeVariant(std::ostream& o, Variant const& v)
{
  if constexpr (isArray<V>() || isArray2D<V>()) {
    using type = typename variant_array_element_type<V>::type;
    rapidjson::OStreamWrapper osw(o);
    rapidjson::Writer<rapidjson::OStreamWrapper> w(osw);

    auto writeArray = [&](auto* values, size_t size) {
      using T = std::remove_pointer_t<decltype(values)>;
      w.StartArray();
      for (auto i = 0u; i < size; ++i) {
        if constexpr (std::is_same_v<int, T>) {
          w.Int(values[i]);
        } else if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
          w.Double(values[i]);
        } else if constexpr (std::is_same_v<int, bool>) {
          w.Bool(values[i]);
        } else if constexpr (std::is_same_v<std::string, T>) {
          w.String(values[i].c_str());
        }
      }
      w.EndArray();
    };

    auto writeArray2D = [&](auto&& array2d) {
      using T = typename std::decay_t<decltype(array2d)>::element_t;
      w.StartArray();
      for (auto i = 0u; i < array2d.rows; ++i) {
        w.StartArray();
        for (auto j = 0u; j < array2d.cols; ++j) {
          if constexpr (std::is_same_v<int, T>) {
            w.Int(array2d(i, j));
          } else if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T>) {
            w.Double(array2d(i, j));
          }
        }
        w.EndArray();
      }
      w.EndArray();
    };

    w.StartObject();
    w.Key("values");
    if constexpr (isArray<V>()) {
      writeArray(v.get<type*>(), v.size());
    } else if constexpr (isArray2D<V>()) {
      writeArray2D(v.get<Array2D<type>>());
    }
    w.EndObject();
  }
}
} // namespace

struct VariantJSONHelpers {
  template <VariantType V>
  static Variant read(std::istream& s)
  {
    rapidjson::Reader reader;
    rapidjson::IStreamWrapper isw(s);
    VariantReader<V> vreader;
    bool ok = reader.Parse(isw, vreader);
    if (ok == false) {
      std::stringstream error;
      error << "Cannot parse serialized Variant, error: " << rapidjson::GetParseError_En(reader.GetParseErrorCode()) << " at offset: " << reader.GetErrorOffset();
      throw std::runtime_error(error.str());
    }
    return vreader.result;
  }

  static void write(std::ostream& o, Variant const& v)
  {
    switch (v.type()) {
      case VariantType::ArrayInt:
        writeVariant<VariantType::ArrayInt>(o, v);
        break;
      case VariantType::ArrayFloat:
        writeVariant<VariantType::ArrayFloat>(o, v);
        break;
      case VariantType::ArrayDouble:
        writeVariant<VariantType::ArrayDouble>(o, v);
        break;
      case VariantType::ArrayBool:
        throw std::runtime_error("Bool vectors not implemented yet");
        //        writeVariant<VariantType::ArrayBool>(o, v);
        break;
      case VariantType::ArrayString:
        writeVariant<VariantType::ArrayString>(o, v);
        break;
      case VariantType::Array2DInt:
        writeVariant<VariantType::Array2DInt>(o, v);
        break;
      case VariantType::Array2DFloat:
        writeVariant<VariantType::Array2DFloat>(o, v);
        break;
      case VariantType::Array2DDouble:
        writeVariant<VariantType::Array2DDouble>(o, v);
        break;
      default:
        break;
    }
  }
};
}

#endif // FRAMEWORK_VARIANTJSONHELPERS_H
