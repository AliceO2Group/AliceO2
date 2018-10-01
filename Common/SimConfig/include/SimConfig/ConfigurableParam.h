// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//first version 8/2018, Sandro Wenzel

#ifndef COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAM_H_
#define COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAM_H_

#include <vector>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <typeinfo>

namespace o2
{
namespace conf
{
// Base class for a configurable parameter.
//
// A configurable parameter is a simple class, defining
// a few (pod) properties/members which are registered
// in a global (boost) property tree.
//
// The features that we provide here are:
// a) Automatic translation from C++ data description to INI/JSON/XML
//    format via ROOT introspection and boost property trees and
//    the possibility to readably save the configuration
// b) Automatic integration of sub-classes into a common configuration
// c) Be able to query properties from high level interfaces (just knowing
// d) Be able to set properties from high-level interfaces (and modifying the underlying
//    C++ object)
// e) Automatic ability to modify parameters from the command-line
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
class ConfigurableParam
{
 public:
  //
  virtual std::string getName() = 0; // print the name of the configurable Parameter

  // print the current keys and values to screen
  virtual void printKeyValues() = 0;

  static void printAllRegisteredParamNames();
  static void printAllKeyValuePairs();

  static void writeJSON(std::string filename);
  static void writeINI(std::string filename);

  // can be used instead of using API on concrete child classes
  template <typename T>
  static T getValueAs(std::string key)
  {
    if (!sIsFullyInitialized) {
      initialize();
    }
    return sPtree->get<T>(key);
  }

  template <typename T>
  static void setValue(std::string mainkey, std::string subkey, T x)
  {
    auto key = mainkey + "." + subkey;
    if (sPtree->get_optional<std::string>(key).is_initialized()) {
      sPtree->put(key, x);
      updateThroughStorageMap(mainkey, subkey, typeid(T), (void*)&x);
    }
  }

  // specialized for std::string
  // which means that the type will be converted internally
  static void setValue(std::string key, std::string valuestring)
  {
    if (sPtree->get_optional<std::string>(key).is_initialized()) {
      sPtree->put(key, valuestring);
      updateThroughStorageMapWithConversion(key, valuestring);
    }
  }

  static void initialize();

  // allows to provide a file from which to update
  // (certain) key-values
  // propagates changes down to each registered configuration
  static void updateFromFile(std::string filename);

  // allows to provide a string of key-values from which to update
  // (certain) key-values
  // propagates changes down to each registered configuration
  // might be useful to get stuff from the command line
  static void updateFromString(std::string);

 protected:
  // constructor is doing nothing else but
  // registering the concrete parameters
  ConfigurableParam();

  static void updatePropertyTree();
  static void updateThroughStorageMap(std::string, std::string, std::type_info const&, void*);
  static void updateThroughStorageMapWithConversion(std::string, std::string);

  virtual ~ConfigurableParam() = default;

  // fill property tree with the key-values from the sub-classes
  virtual void putKeyValues(boost::property_tree::ptree*) = 0;

  // static map keeping, for each configuration key, its memory location and type
  // (internal use to easily sync updates, this is ok since parameter classes are singletons)
  static std::map<std::string, std::pair<int, void*>>* sKeyToStorageMap;

 private:
  // static registry for implementations of this type
  static std::vector<ConfigurableParam*>* sRegisteredParamClasses; //!
  // static property tree (stocking all key - value pairs from instances of type ConfigurableParam)
  static boost::property_tree::ptree* sPtree; //!
  static bool sIsFullyInitialized;            //!
};

} // end namespace conf
} // end namespace o2

// a helper macro for boilerplate code in parameter classes
#define O2ParamDef(classname, key)               \
 private:                                        \
  static constexpr char const* const sKey = key; \
  static classname sInstance;                    \
  classname() = default;                         \
  template <typename T>                          \
  friend class o2::conf::ConfigurableParamHelper;

// a helper macro to implement necessary symbols in source
#define O2ParamImpl(classname) classname classname::sInstance;

#endif /* COMMON_SIMCONFIG_INCLUDE_SIMCONFIG_CONFIGURABLEPARAM_H_ */
