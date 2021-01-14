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
    : states{},
      rows{0},
      cols{0}
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
    if constexpr (!std::is_same_v<int, variant_array_element_type_t<V>>) {
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
    if constexpr (!(std::is_same_v<float, variant_array_element_type_t<V>> || std::is_same_v<double, variant_array_element_type_t<V>>)) {
      states.push(State::IN_ERROR);
      return true;
    }
    if (states.top() == State::IN_ARRAY || states.top() == State::IN_ROW) {
      if constexpr (std::is_same_v<double, variant_array_element_type_t<V>>) {
        debug << "added to array as double" << std::endl;
        accumulatedData.push_back(d);
        return true;
      } else if constexpr (std::is_same_v<float, variant_array_element_type_t<V>>) {
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
    if constexpr (!std::is_same_v<bool, variant_array_element_type_t<V>>) {
      states.push(State::IN_ERROR);
      return false;
    } else {
      if (states.top() == State::IN_ARRAY) {
        debug << "added to array" << std::endl;
        accumulatedData.push_back(b);
        return true;
      }
      states.push(State::IN_ERROR);
      return true;
    }
  }

  bool String(const Ch* str, SizeType, bool)
  {
    debug << "String(" << str << ")" << std::endl;
    if (states.top() == State::IN_ERROR) {
      debug << "In ERROR state" << std::endl;
      return false;
    }
    if constexpr (!(V == VariantType::ArrayString || isLabeledArray<V>())) {
      states.push(State::IN_ERROR);
      return true;
    } else {
      if (states.top() == State::IN_ARRAY) {
        debug << "added to array" << std::endl;
        if constexpr (isLabeledArray<V>()) {
          if (currentKey == labels_rows_str) {
            labels_rows.push_back(str);
            return true;
          } else if (currentKey == labels_cols_str) {
            labels_cols.push_back(str);
            return true;
          } else {
            states.push(State::IN_ERROR);
            return true;
          }
        } else {
          accumulatedData.push_back(str);
        }
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
      currentKey = str;
      return false;
    }
    if (states.top() == State::IN_DATA) {
      //no previous keys
      states.push(State::IN_KEY);
      currentKey = str;
      return true;
    }
    if (states.top() == State::IN_KEY) {
      currentKey = str;
      if constexpr (!isLabeledArray<V>()) {
        debug << "extra keys in a single-key variant" << std::endl;
        states.push(State::IN_ERROR);
        return true;
      }
      return true;
    }
    currentKey = str;
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
      if constexpr (isArray<V>()) {
        debug << "creating 1d-array variant" << std::endl;
        result = Variant(accumulatedData);
      } else if constexpr (isArray2D<V>()) {
        debug << "creating 2d-array variant" << std::endl;
        assert(accumulatedData.size() == rows * cols);
        result = Variant(Array2D{accumulatedData, rows, cols});
      } else if constexpr (isLabeledArray<V>()) {
        debug << "creating labeled array variant" << std::endl;
        assert(accumulatedData.size() == rows * cols);
        if (labels_rows.empty() == false) {
          assert(labels_rows.size() == rows);
        }
        if (labels_cols.empty() == false) {
          assert(labels_cols.size() == cols);
        }
        result = Variant(LabeledArray{Array2D{accumulatedData, rows, cols}, labels_rows, labels_cols});
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
      return true;
    } else if (states.top() == State::IN_ARRAY) {
      if constexpr (isArray2D<V>() || isLabeledArray<V>()) {
        states.push(State::IN_ROW);
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
      if constexpr (isArray2D<V>() || isLabeledArray<V>()) {
        rows = elementCount;
      }
      return true;
    } else if (states.top() == State::IN_ROW) {
      //finish up row
      states.pop();
      if constexpr (isArray2D<V>() || isLabeledArray<V>()) {
        cols = elementCount;
      }
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
  std::vector<variant_array_element_type_t<V>> accumulatedData;
  std::vector<std::string> labels_rows;
  std::vector<std::string> labels_cols;
  Variant result;
};

template <VariantType V>
void writeVariant(std::ostream& o, Variant const& v)
{
  if constexpr (isArray<V>() || isArray2D<V>() || isLabeledArray<V>()) {
    using type = variant_array_element_type_t<V>;
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
        } else if constexpr (std::is_same_v<bool, T>) {
          w.Bool(values[i]);
        } else if constexpr (std::is_same_v<std::string, T>) {
          w.String(values[i].c_str());
        }
      }
      w.EndArray();
    };

    auto writeVector = [&](auto&& vector) {
      return writeArray(vector.data(), vector.size());
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

    auto writeLabeledArray = [&](auto&& array) {
      w.Key(labels_rows_str);
      writeVector(array.getLabelsRows());
      w.Key(labels_cols_str);
      writeVector(array.getLabelsCols());
      w.Key("values");
      writeArray2D(array.getData());
    };

    w.StartObject();
    if constexpr (isArray<V>()) {
      w.Key("values");
      writeArray(v.get<type*>(), v.size());
    } else if constexpr (isArray2D<V>()) {
      w.Key("values");
      writeArray2D(v.get<Array2D<type>>());
    } else if constexpr (isLabeledArray<V>()) {
      writeLabeledArray(v.get<LabeledArray<type>>());
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
      case VariantType::LabeledArrayInt:
        writeVariant<VariantType::LabeledArrayInt>(o, v);
        break;
      case VariantType::LabeledArrayFloat:
        writeVariant<VariantType::LabeledArrayFloat>(o, v);
        break;
      case VariantType::LabeledArrayDouble:
        writeVariant<VariantType::LabeledArrayDouble>(o, v);
        break;
      default:
        break;
    }
  }
};
}

#endif // FRAMEWORK_VARIANTJSONHELPERS_H
