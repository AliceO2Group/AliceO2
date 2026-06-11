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

// first version 8/2018, Sandro Wenzel

#ifndef COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAM_H_
#define COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAM_H_

#include <algorithm>
#include <cassert>
#include <cctype>
#include <concepts>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <boost/property_tree/ptree_fwd.hpp>
#include <typeinfo>
#include <iostream>
#include <array>

class TFile;
class TRootIOCtor;
class TDataMember;

namespace o2
{
namespace conf
{
// Base class for a configurable parameter.
//
// A configurable parameter (ConfigurableParameter) is a simple class, defining
// a few (pod) properties/members which are registered
// in a global (boost) property tree / structure.
//
// The features that we provide here are:
// *) Automatic translation from C++ data description to INI/JSON/XML
//    format via ROOT introspection and boost property trees and
//    the possibility to readably save the configuration
// *) Serialization/Deserialization into ROOT binary blobs (for the purpose
//    of writing/retrieving parameters from CCDB and to pass parameters along the processing chain)
// *) Automatic integration of sub-classes into a common configuration
// *) Be able to query properties from high level interfaces (just knowing
// *) Be able to set properties from high-level interfaces (and modifying the underlying
//    C++ object)
// *) Automatic ability to modify parameters from the command-line
// *) Keeping track of the provenance of individual parameter values; The user is able
//    to see whether is parameter is defaulted-code-value/coming-from-CCDB/coming-from-comandline
//
// Note that concrete parameter sub-classes **must** be implemented
// by inheriting from ConfigurableParamHelper and not from this class.
//
// ---------------------
// Example: To define a parameter class TPCGasParameters, one does the following:
//
// class TPCGasParamer : public ConfigurableParamHelper<TPCGasParameter>
// {
//   public:
//     double getGasDensity() const { return mGasDensity; }
//   private: // define properties AND default values
//     double mGasDensity = 1.23;
//     int mGasMaterialID = 1;
//
//     O2ParamDef(TPCGasParameter, TPCGas); // a macro implementing some magic
// }
//
//
// We can now query the parameters in various ways
// - All parameter classes are singletons and we can say: TPCGasParameter::Instance().getGasDensity();
// - We can query by key (using classname + parameter name) from the global registry:
// -    ConfigurableParameter::getValueAs<double>("TPCGas", "mGasDensity");
//
// We can modify the parameters via the global registry together with an automatic syncing
// of the underlying C++ object:
// - ConfigurableParameter::setValue("TPCGas.mGasDensity", "0.5");
//
// - TPCGasParameter::Instance().getGasParameter() will now return 0.5;
//
// This feature allows to easily modify parameters at run-time via a textual representation
// (for example by giving strings on the command line)
//
// The collection of all parameter keys and values can be stored to a human/machine readable
// file
//  - ConfigurableParameter::writeJSON("thisconfiguration.json")

struct EnumLegalValues {
  std::vector<std::pair<std::string, int>> vvalues;

  bool isLegal(const std::string& value) const
  {
    for (auto& v : vvalues) {
      if (v.first == value) {
        return true;
      }
    }
    return false;
  }

  bool isLegal(int value) const
  {
    for (auto& v : vvalues) {
      if (v.second == value) {
        return true;
      }
    }
    return false;
  }

  std::string toString() const;
  int getIntValue(const std::string& value) const;
};

class EnumRegistry
{
 public:
  void add(const std::string& key, const TDataMember* dm);

  bool contains(const std::string& key) const
  {
    return entries.count(key) > 0;
  }

  std::string toString() const;

  const EnumLegalValues* operator[](const std::string& key) const
  {
    auto iter = entries.find(key);
    return iter != entries.end() ? &iter->second : nullptr;
  }

 private:
  std::unordered_map<std::string, EnumLegalValues> entries;
};

template <typename T>
concept Container = !std::is_same_v<std::remove_cvref_t<T>, std::string> && requires(T t) {
  typename T::value_type;
  typename T::iterator;
  { t.begin() } -> std::same_as<typename T::iterator>;
  { t.end() } -> std::same_as<typename T::iterator>;
};

template <typename T>
concept MapLike = Container<T> && requires {
  typename T::key_type;
  typename T::mapped_type;
};

template <typename T>
concept SequenceContainer = Container<T> && !MapLike<T>;

template <typename T>
concept HasPushBack = requires(T a, typename T::value_type v) {
  { a.push_back(v) } -> std::same_as<void>;
};

template <typename>
inline constexpr bool AlwaysFalse = false;

class ContainerParser
{
 public:
  template <typename T>
  static T parse(const std::string& str)
  {
    if constexpr (MapLike<T>) {
      return parseMap<T>(str);
    } else if constexpr (SequenceContainer<T>) {
      return parseSet<T>(str);
    } else if constexpr (Container<T>) {
      return parseSequence<T>(str);
    } else {
      return parseScalar<T>(str);
    }
  }

  static std::string trim(const std::string& str)
  {
    auto start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
      return "";
    }
    auto end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
  }

 private:
  // Parse vector, list, deque, array
  template <SequenceContainer SequenceT>
  static SequenceT parseSequence(const std::string& str)
  {
    SequenceT result;
    using ValueType = typename SequenceT::value_type;
    std::string cleaned = str;
    if (!cleaned.empty() && cleaned.front() == '[' && cleaned.back() == ']') { // removed brackets [1,2,3] -> 1,2,3
      cleaned = cleaned.substr(1, cleaned.length() - 2);
    }
    if (cleaned.empty() || cleaned == "{}") { // nothing to do
      return result;
    }
    if constexpr (Container<ValueType>) {
      static_assert(AlwaysFalse<ValueType>, "Nested containers are not supported as configurable parameters");
    }
    auto tokens = split(cleaned, ',');
    for (const auto& token : tokens) {
      std::string trimmed = trim(token);
      result.insert(result.end(), parseScalar<ValueType>(trimmed));
    }
    return result;
  }

  // Parse map, unordered_map, multimap
  template <MapLike MapT>
  static MapT parseMap(const std::string& str)
  {
    MapT result;
    using KeyType = typename MapT::key_type;
    using ValueType = typename MapT::mapped_type;
    std::string cleaned = str;
    if (!cleaned.empty() && cleaned.front() == '{' && cleaned.back() == '}') { // stip braces {a:1,b:2} -> a:1,b:2
      cleaned = cleaned.substr(1, cleaned.length() - 2);
    }
    if (cleaned.empty()) { // nothing to do
      return result;
    }
    if constexpr (Container<KeyType> || Container<ValueType>) {
      static_assert(AlwaysFalse<MapT>, "Nested containers are not supported as configurable parameters");
    }
    auto pairs = split(cleaned, ',');
    for (const auto& pair_str : pairs) {
      auto kv = split(pair_str, ':');
      if (kv.size() != 2) {
        throw std::runtime_error("Invalid map syntax: " + pair_str + ". Expected 'key:value' format, got ");
      }
      KeyType key = parseScalar<KeyType>(trim(kv[0]));
      result[key] = parseScalar<ValueType>(trim(kv[1]));
    }
    return result;
  }

  // Parse set containers
  template <SequenceContainer SetT>
  static SetT parseSet(const std::string& str)
  {
    SetT result;
    using ValueType = typename SetT::value_type;
    auto vec = parseSequence<std::vector<ValueType>>(str);
    for (const auto& val : vec) {
      if constexpr (HasPushBack<SetT>) {
        result.push_back(val);
      } else {
        result.insert(val);
      }
    }
    return result;
  }

  // Parse scalar types
  template <typename T>
  static T parseScalar(const std::string& str)
  {
    if constexpr (std::is_same_v<T, std::string>) {
      return str;
    } else if constexpr (std::is_same_v<T, bool>) {
      std::string lower = str;
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (lower == "true" || lower == "1") {
        return true;
      }
      if (lower == "false" || lower == "0") {
        return false;
      }
      throw std::runtime_error("Invalid boolean value: " + str);
    } else if constexpr (std::is_same_v<T, char> || std::is_same_v<T, signed char>) {
      size_t pos = 0;
      long long value = std::stoll(str, &pos);
      if (pos != str.size()) {
        throw std::runtime_error("Failed to parse '" + str + "' as char type");
      }
      if (value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max()) {
        throw std::runtime_error("Value out of range for char type: " + str);
      }
      return static_cast<T>(value);
    } else if constexpr (std::is_same_v<T, unsigned char>) {
      size_t pos = 0;
      unsigned long long value = std::stoull(str, &pos);
      if (pos != str.size()) {
        throw std::runtime_error("Failed to parse '" + str + "' as unsigned char type");
      }
      if (value > std::numeric_limits<T>::max()) {
        throw std::runtime_error("Value out of range for unsigned char type: " + str);
      }
      return static_cast<T>(value);
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
      if (!str.empty() && str.front() == '-') {
        throw std::runtime_error("Value out of range for unsigned integer type: " + str);
      }
      size_t pos = 0;
      unsigned long long value = std::stoull(str, &pos);
      if (pos != str.size() || value > std::numeric_limits<T>::max()) {
        throw std::runtime_error("Failed to parse '" + str + "' as unsigned integer type");
      }
      return static_cast<T>(value);
    } else if constexpr (std::is_integral_v<T>) {
      size_t pos = 0;
      long long value = std::stoll(str, &pos);
      if (pos != str.size() || value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max()) {
        throw std::runtime_error("Failed to parse '" + str + "' as signed integer type");
      }
      return static_cast<T>(value);
    } else if constexpr (std::is_floating_point_v<T>) {
      size_t pos = 0;
      long double value = std::stold(str, &pos);
      if (pos != str.size()) {
        throw std::runtime_error("Failed to parse '" + str + "' as floating point type");
      }
      return static_cast<T>(value);
    } else {
      std::istringstream iss(str);
      T value;
      iss >> value;
      iss >> std::ws;
      if (iss.fail() || !iss.eof()) {
        throw std::runtime_error("Failed to parse '" + str + "' as " + typeid(T).name());
      }
      return value;
    }
  }

  // Split respecting nested brackets and braces
  static std::vector<std::string> split(const std::string& str, char delimiter)
  {
    std::vector<std::string> tokens;
    std::string current;
    int bracket_depth = 0;
    int brace_depth = 0;
    for (char c : str) {
      if (c == '[') {
        bracket_depth++;
      } else if (c == ']') {
        bracket_depth--;
      } else if (c == '{') {
        brace_depth++;
      } else if (c == '}') {
        brace_depth--;
      } else if (c == delimiter && bracket_depth == 0 && brace_depth == 0) {
        if (!current.empty()) {
          tokens.push_back(current);
          current.clear();
        }
        continue;
      }
      current += c;
    }
    if (!current.empty()) {
      tokens.push_back(current);
    }
    return tokens;
  }
};

class ConfigurableParam
{
 public:
  enum EParamProvenance {
    kCODE /* from default code initialization */,
    kCCDB /* overwritten from CCDB */,
    kRT /* changed during runtime via API call setValue (for example command line) */
    /* can add more modes here */
  };

  enum class EParamUpdateStatus {
    Changed,   // param was successfully changed
    Unchanged, // param was not changed: new value is the same as previous
    Failed     // failed to update param
  };

  static std::string toString(EParamProvenance p)
  {
    static std::array<std::string, 3> names = {"CODE", "CCDB", "RT"};
    return names[(int)p];
  }

  // get the name of the configurable Parameter
  virtual std::string getName() const = 0;

  // print the current keys and values to screen (optionally with provenance information)
  virtual void printKeyValues(bool showprov = true, bool useLogger = false, bool withPadding = false, bool showHash = false) const = 0;

  // get a single size_t hash_value of this parameter (can be used as a checksum to see
  // if object changed or different)
  virtual size_t getHash() const = 0;

  // return the provenance of the member key
  virtual EParamProvenance getMemberProvenance(const std::string& key) const = 0;

  static EParamProvenance getProvenance(const std::string& key);

  static void printAllRegisteredParamNames();
  static void printAllKeyValuePairs(bool useLogger = false);

  static const std::string& getOutputDir() { return sOutputDir; }

  static void setOutputDir(const std::string& d) { sOutputDir = d; }

  static bool configFileExists(std::string const& filepath);

  // writes a human readable JSON file of all parameters
  static void writeJSON(std::string const& filename, std::string const& keyOnly = "");
  // writes a human readable INI file of all parameters
  static void writeINI(std::string const& filename, std::string const& keyOnly = "");

  // writes a human readable INI or JSON file depending on the extension
  static void write(std::string const& filename, std::string const& keyOnly = "");

  // can be used instead of using API on concrete child classes
  template <typename T>
  static T getValueAs(std::string key)
  {
    return [](auto* tree, const std::string& key) -> T {
      if (!sIsFullyInitialized) {
        initialize();
      }
      return tree->template get<T>(key);
    }(sPtree, key);
  }

  template <typename T>
  static void setValue(std::string const& mainkey, std::string const& subkey, T x)
  {
    if (!sIsFullyInitialized) {
      initialize();
    }
    return [&subkey, &x, &mainkey](auto* tree) -> void {
      assert(tree);
      try {
        auto key = mainkey + "." + subkey;
        if (tree->template get_optional<std::string>(key).is_initialized()) {
          tree->put(key, x);
          auto changed = updateThroughStorageMap(mainkey, subkey, typeid(T), (void*)&x);
          if (changed != EParamUpdateStatus::Failed) {
            sValueProvenanceMap->find(key)->second = kRT; // set to runtime
          }
        }
      } catch (std::exception const& e) {
        std::cerr << "Error in setValue (T) " << e.what() << "\n";
      }
    }(sPtree);
  }

  static void setProvenance(std::string const& mainkey, std::string const& subkey, EParamProvenance p)
  {
    if (!sIsFullyInitialized) {
      std::cerr << "setProvenance was called on non-initialized ConfigurableParam\n";
      return;
    }
    try {
      auto key = mainkey + "." + subkey;
      auto keyProv = sValueProvenanceMap->find(key);
      if (keyProv != sValueProvenanceMap->end()) {
        keyProv->second = p;
      }
    } catch (std::exception const& e) {
      std::cerr << "Error in setProvenance (T) " << e.what() << "\n";
    }
  }

  // specialized for std::string
  // which means that the type will be converted internally
  static void setValue(std::string const& key, std::string const& valuestring);
  static void setEnumValue(const std::string&, const std::string&);
  static void setArrayValue(const std::string&, const std::string&);
  static void setContainerValue(const std::string&, const std::string&);
  static bool isRegisteredContainerType(const std::string& typeName);
  static void registerContainerType(const std::string& key, const std::string& typeName);
  static std::string getRegisteredContainerType(const std::string& key);
  static bool assignRegisteredContainer(const std::string& typeName, void* target, const void* source);
  static bool areRegisteredContainersEqual(const std::string& typeName, const void* lhs, const void* rhs);
  static std::string registeredContainerAsString(const std::string& typeName, const void* source);

  // update the storagemap from a vector of key/value pairs, calling setValue for each pair
  static void setValues(std::vector<std::pair<std::string, std::string>> const& keyValues);

  // initializes the parameter database
  static void initialize();

  // create CCDB snapshot
  static void toCCDB(std::string filename);
  // load from (CCDB) snapshot
  static void fromCCDB(std::string filename);

  // allows to provide a string of key-values from which to update
  // (certain) key-values
  // propagates changes down to each registered configuration
  // might be useful to get stuff from the command line
  static void updateFromString(std::string const&);

  // provide a path to a configuration file with ConfigurableParam key/values
  // If nonempty comma-separated paramsList is provided, only those params will
  // be updated, absence of data for any of requested params will lead to fatal
  static void updateFromFile(std::string const&, std::string const& paramsList = "", bool unchangedOnly = false);

  // interface for use from the CCDB API; allows to sync objects read from CCDB with the information
  // stored in the registry; modifies given object as well as registry
  virtual void syncCCDBandRegistry(void* obj) = 0;

 protected:
  // constructor is doing nothing else but
  // registering the concrete parameters
  ConfigurableParam();

  friend std::ostream& operator<<(std::ostream& out, const ConfigurableParam& me);

  static void initPropertyTree();
  static EParamUpdateStatus updateThroughStorageMap(std::string, std::string, std::type_info const&, void*);
  static EParamUpdateStatus updateThroughStorageMapWithConversion(std::string const&, std::string const&);

  virtual ~ConfigurableParam() = default;

  // fill property tree with the key-values from the sub-classes
  virtual void putKeyValues(boost::property_tree::ptree*) = 0;
  virtual void output(std::ostream& out) const = 0;

  virtual void serializeTo(TFile*) const = 0;
  virtual void initFrom(TFile*) = 0;

  // static map keeping, for each configuration key, its memory location and type
  // (internal use to easily sync updates, this is ok since parameter classes are singletons)
  static std::map<std::string, std::pair<std::type_info const&, void*>>* sKeyToStorageMap;

  // keep track of provenance of parameters and values
  static std::map<std::string, ConfigurableParam::EParamProvenance>* sValueProvenanceMap;

  // A registry of enum names and their allowed values
  // (stored as a vector of pairs <enumValueLabel, enumValueInt>)
  static EnumRegistry* sEnumRegistry;

  static std::string sOutputDir;

  void setRegisterMode(bool b) { sRegisterMode = b; }
  bool isInitialized() const { return sIsFullyInitialized; }

  // friend class o2::ccdb::CcdbApi;
 private:
  // static registry for implementations of this type
  static std::vector<ConfigurableParam*>* sRegisteredParamClasses; //!
  // static property tree (stocking all key - value pairs from instances of type ConfigurableParam)
  static boost::property_tree::ptree* sPtree; //!
  static bool sIsFullyInitialized;            //!
  static bool sRegisterMode;                  //! (flag to enable/disable autoregistering of child classes)
};

} // end namespace conf
} // end namespace o2

// a helper macro for boilerplate code in parameter classes
#define O2ParamDef(classname, key)                \
 public:                                          \
  classname(TRootIOCtor*) {}                      \
  classname(classname const&) = delete;           \
                                                  \
 private:                                         \
  static constexpr char const* const sKey = key;  \
  static classname sInstance;                     \
  classname() = default;                          \
  template <typename T>                           \
  friend class o2::conf::ConfigurableParamHelper; \
  template <typename T, typename P>               \
  friend class o2::conf::ConfigurableParamPromoter;

// a helper macro to implement necessary symbols in source
#define O2ParamImpl(classname) classname classname::sInstance;

#endif /* COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAM_H_ */
