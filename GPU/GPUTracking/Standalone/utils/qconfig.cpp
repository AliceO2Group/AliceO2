// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfig.cpp
/// \author David Rohr

#include <cstdio>
#include <cstdlib>
#include <memory.h>
#include <utility>
#include <vector>
#include <functional>
#include <iostream>
#include <tuple>
#include "qconfig.h"

// Create config instances
#define QCONFIG_INSTANCE
#include "qconfig.h"
#undef QCONFIG_INSTANCE

namespace qConfig
{
#define qon_mcat(a, b) a##b
#define qon_mxcat(a, b) qon_mcat(a, b)
#define qon_mcat3(a, b, c) a##b##c
#define qon_mxcat3(a, b, c) qon_mcat3(a, b, c)
#define qon_mstr(a) #a
#define qon_mxstr(a) qon_mstr(a)
#define QCONFIG_SETTING(name, type)                     \
  struct qon_mxcat3(q, name, _t)                        \
  {                                                     \
    type v;                                             \
    constexpr qon_mxcat3(q, name, _t)(type s) : v(s) {} \
  };                                                    \
  constexpr qon_mxcat3(q, name, _t) name(type v) { return (qon_mxcat3(q, name, _t)(v)); }

#define QCONFIG_SETTING_TEMPLATE(name)                      \
  template <typename T>                                     \
  struct qon_mxcat3(q, name, _t)                            \
  {                                                         \
    T v;                                                    \
    constexpr qon_mxcat3(q, name, _t)(const T& s) : v(s) {} \
  };                                                        \
  template <typename T>                                     \
  constexpr qon_mxcat3(q, name, _t)<T> name(T v)            \
  {                                                         \
    return (qon_mxcat3(q, name, _t) < T > (v));             \
  }

QCONFIG_SETTING(message, const char*)
QCONFIG_SETTING_TEMPLATE(min)
QCONFIG_SETTING_TEMPLATE(max)
QCONFIG_SETTING_TEMPLATE(set)
QCONFIG_SETTING_TEMPLATE(def)

static inline const char* getOptName(const char** argv, int i)
{
  while (i > 1 && argv[i][0] != '-') {
    i--;
  }
  return (argv[i]);
}

template <typename T>
struct qConfigSettings {
  qConfigSettings() : checkMin(false), checkMax(false), doSet(false), doDefault(false), min(0), max(0), set(0), message(nullptr), allowEmpty(false){};
  template <typename S>
  qConfigSettings(const qConfigSettings<S> v) : checkMin(false), checkMax(false), doSet(false), doDefault(false), min(0), max(0), set(0), message(v.message), allowEmpty(v.allowEmpty){};
  bool checkMin, checkMax;
  bool doSet, doDefault;
  T min, max;
  T set;
  const char* message;
  bool allowEmpty;
};

template <typename T>
static int qAddOptionType(qConfigSettings<T>& settings, T& ref, int& i, const char** argv, const int argc, T def);
template <typename T>
static void qAddOptionMessage(qConfigSettings<T>& settings, T& ref);
template <typename T>
static int qAddOptionMinMax(qConfigSettings<T>& settings, T& ref, const char* arg);

template <typename T>
struct qSettingsType {
  typedef T settingsType;
};
template <typename... X>
struct qSettingsType<std::tuple<X...>> {
  typedef int settingsType;
};

template <typename T>
struct qConfigTypeSpecialized {
  static inline int qAddOptionMain(qConfigSettings<T>& settings, T& ref, int& i, const char** argv, const int argc, T def);
};
template <typename... X>
struct qConfigTypeSpecialized<std::tuple<X...>> {
  static inline int qAddOptionMain(qConfigSettings<typename qSettingsType<std::tuple<X...>>::settingsType>& settings, std::tuple<X...>& ref, int& i, const char** argv, const int argc, std::tuple<X...>& def);
};

// Main processing function for arguments
template <class T>
inline int qConfigTypeSpecialized<T>::qAddOptionMain(qConfigSettings<T>& settings, T& ref, int& i, const char** argv, const int argc, T def)
{
  int retVal = 0;
  int iOrg = i;
  if (settings.doSet) {
    ref = settings.set;
  } else if ((retVal = qAddOptionType<T>(settings, ref, i, argv, argc, def))) {
    return (retVal);
  }
  if ((retVal = qAddOptionMinMax<T>(settings, ref, iOrg < argc ? argv[iOrg] : ""))) {
    return (retVal);
  }
  qAddOptionMessage<T>(settings, ref);
  return (0);
}
template <typename T>
static inline int qAddOptionMainTupleElem(qConfigSettings<typename qSettingsType<T>::settingsType> settingsTup, T& ref, int& i, const char** argv, const int argc)
{
  T def = T();
  qConfigSettings<T> settings = settingsTup;
  return (qAddOptionType<T>(settings, ref, i, argv, argc, def));
}
template <typename T, int index = 0, int left = std::tuple_size<T>::value>
struct qAddOptionMainTupleStruct {
  static inline int qAddOptionMainTuple(qConfigSettings<typename qSettingsType<T>::settingsType> settings, T& tup, int& i, const char** argv, const int argc)
  {
    auto& ref = std::get<index>(tup);
    int retVal = qAddOptionMainTupleElem(settings, ref, i, argv, argc);
    if (retVal) {
      if (retVal == qcrArgMissing && index != 0) {
        printf("Invalid number of arguments for option %s\n", getOptName(argv, i));
        retVal = qcrArgIncomplete;
      }
      return (retVal);
    }
    return (qAddOptionMainTupleStruct<T, index + 1, left - 1>::qAddOptionMainTuple(settings, tup, i, argv, argc));
  }
};
template <typename T, int index>
struct qAddOptionMainTupleStruct<T, index, 0> {
  static inline int qAddOptionMainTuple(qConfigSettings<typename qSettingsType<T>::settingsType> /*settings*/, T& /*tup*/, int& /*i*/, const char** /*argv*/, const int /*argc*/) { return 0; }
};
template <typename... X>
inline int qConfigTypeSpecialized<std::tuple<X...>>::qAddOptionMain(qConfigSettings<typename qSettingsType<std::tuple<X...>>::settingsType>& settings, std::tuple<X...>& ref, int& i, const char** argv, const int argc, std::tuple<X...>& /*def*/)
{
  return (qAddOptionMainTupleStruct<std::tuple<X...>>::qAddOptionMainTuple(settings, ref, i, argv, argc));
}

template <typename T>
struct qConfigType {
  // Recursive handling of additional settings
  static inline void qProcessSetting(qConfigSettings<T>& settings, qmin_t<T> minval)
  {
    static_assert(!std::is_same<T, bool>::value, "min option not supported for boolean settings");
    settings.checkMin = true;
    settings.min = minval.v;
  }
  static inline void qProcessSetting(qConfigSettings<T>& settings, qmax_t<T> maxval)
  {
    static_assert(!std::is_same<T, bool>::value, "max option not supported for boolean settings");
    settings.checkMax = true;
    settings.max = maxval.v;
  }
  static inline void qProcessSetting(qConfigSettings<T>& settings, qmessage_t msg) { settings.message = msg.v; }
  static inline void qProcessSetting(qConfigSettings<T>& settings, qset_t<T> set)
  {
    settings.doSet = true;
    settings.set = set.v;
  }
  static inline void qProcessSetting(qConfigSettings<T>& settings, qdef_t<T> set)
  {
    settings.doDefault = true;
    settings.set = set.v;
  }

  static inline void qAddOptionSettings(qConfigSettings<T>& /*settings*/) {}
  template <typename Arg1, typename... Args>
  static inline void qAddOptionSettings(qConfigSettings<T>& settings, Arg1&& arg1, Args&&... args)
  {
    qProcessSetting(settings, arg1);
    qAddOptionSettings(settings, args...);
  }

  template <typename... Args>
  static inline int qAddOption(T& ref, int& i, const char** argv, const int argc, T def, const char* /*help*/, Args&&... args)
  {
    auto settings = qConfigGetSettings(args...);
    return qConfigTypeSpecialized<T>::qAddOptionMain(settings, ref, i, argv, argc, def);
  }

  template <typename... Args>
  static inline int qAddOptionVec(std::vector<T>& ref, int& i, const char** argv, const int argc, const char* /*help*/, Args&&... args)
  {
    auto settings = qConfigGetSettings(args...);
    int iFirst = i, iLast;
    do {
      iLast = i;
      T tmp = T(), def = T();
      settings.allowEmpty = (i != iFirst);
      int retVal = qConfigTypeSpecialized<T>::qAddOptionMain(settings, tmp, i, argv, argc, def);
      if (retVal) {
        return (retVal != qcrArgMissing || i == iFirst ? retVal : 0);
      }
      if (i == iFirst || i != iLast) {
        ref.push_back(tmp);
      }
    } while (i != iLast);
    return (0);
  }

  template <typename... Args>
  static inline int qAddOptionArray(T* ref, int count, int& i, const char** argv, const int argc, const char* /*help*/, Args&&... args)
  {
    auto settings = qConfigGetSettings(args...);
    int iFirst = i, iLast;
    do {
      iLast = i;
      T tmp = T(), def = T();
      settings.allowEmpty = (i != iFirst);
      int retVal = qConfigTypeSpecialized<T>::qAddOptionMain(settings, tmp, i, argv, argc, def);
      if (retVal) {
        return (retVal != qcrArgMissing || i == iFirst ? retVal : 0);
      }
      if (i - iFirst >= count) {
        printf("Too many values provided for option %s\n", getOptName(argv, i));
        return (qcrArrayOverflow);
      }
      if (i == iFirst || i != iLast) {
        ref[i - iFirst] = tmp;
      }
    } while (i != iLast);
    return (0);
  }

  template <typename... Args>
  static inline void qConfigHelpOption(const char* name, const char* type, const char* def, const char* optname, char optnameshort, const char* preopt, char preoptshort, int optionType, const char* help, Args&&... args)
  {
    auto settings = qConfigGetSettings(args...);
    const bool boolType = optionType != 1 && std::is_same<T, bool>::value;
    const char* arguments = settings.doSet ? " (" : (settings.doDefault || optionType == 1 || boolType) ? " [arg] (" : optionType == 2 ? " [...] (" : " arg (";
    char argBuffer[4] = { 0 };
    unsigned int argBufferPos = 0;
    if (optnameshort && preoptshort) {
      argBuffer[argBufferPos++] = '-';
    }
    if (optnameshort) {
      argBuffer[argBufferPos++] = preoptshort == 0 ? '-' : preoptshort;
      argBuffer[argBufferPos++] = optnameshort;
    }
    std::cout << "\t" << name << ": " << argBuffer << (optnameshort == 0 ? "" : arguments) << "--" << preopt << optname << (optnameshort == 0 ? arguments : ", ") << "type: " << type;
    if (optionType == 0) {
      std::cout << ", default: " << def;
    }
    if (optionType == 1) {
      std::cout << ", sets " << name << " to " << def;
    }
    if (settings.checkMin) {
      std::cout << ", minimum: " << settings.min;
    }
    if (settings.checkMax) {
      std::cout << ", maximum: " << settings.max;
    }
    std::cout << ")\n\t\t" << help << ".\n";
    if (settings.doSet) {
      std::cout << "\t\tSets " << name << " to " << settings.set << ".\n";
    } else if (settings.doDefault) {
      std::cout << "\t\tIf no argument is supplied, " << name << " is set to " << settings.set << ".\n";
    } else if (boolType) {
      std::cout << "\t\tIf no argument is supplied, " << name << " is set to true.\n";
    }
    if (optionType == 2) {
      std::cout << "\t\tCan be set multiple times, accepts multiple arguments.\n";
    }
    std::cout << "\n";
  }

  template <typename... Args>
  static inline auto qConfigGetSettings(Args&... args)
  {
    qConfigSettings<typename qSettingsType<T>::settingsType> settings;
    qConfigType<typename qSettingsType<T>::settingsType>::qAddOptionSettings(settings, args...);
    return (settings);
  }
};

static inline const char* getArg(int& i, const char** argv, const int argc, bool allowOption = false)
{
  if (i + 1 < argc && argv[i + 1][0] && ((allowOption && argv[i + 1][0] == '-' && argv[i + 1][1] != '-') || argv[i + 1][0] != '-')) {
    return (argv[++i]);
  }
  return (nullptr);
}

template <class T>
static inline int qAddOptionGeneric(qConfigSettings<T>& settings, T& ref, int& i, const char** argv, const int argc, T def, std::function<T(const char*)> func, bool allowDefault = false)
{
  const char* arg = getArg(i, argv, argc, !allowDefault);
  if (arg) {
    ref = func(arg);
    return (0);
  } else if (allowDefault) {
    ref = def;
    return (0);
  }
  if (!settings.allowEmpty) {
    printf("Argument missing for option %s!\n", getOptName(argv, i));
  }
  return (qcrArgMissing);
}

// Handling of all supported types
template <>
inline int qAddOptionType<bool>(qConfigSettings<bool>& settings, bool& ref, int& i, const char** argv, const int argc, bool /*def*/)
{
  return qAddOptionGeneric<bool>(
    settings, ref, i, argv, argc, settings.doDefault ? settings.set : true, [](const char* a) -> bool {
      return atoi(a);
    },
    true);
}
template <>
inline int qAddOptionType<char>(qConfigSettings<char>& settings, char& ref, int& i, const char** argv, const int argc, char /*def*/)
{
  return qAddOptionGeneric<char>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> char {
      return atoi(a);
    },
    settings.doDefault);
}
template <>
inline int qAddOptionType<int>(qConfigSettings<int>& settings, int& ref, int& i, const char** argv, const int argc, int /*def*/)
{
  return qAddOptionGeneric<int>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> int {
      return atoi(a);
    },
    settings.doDefault);
}
template <>
inline int qAddOptionType<unsigned int>(qConfigSettings<unsigned int>& settings, unsigned int& ref, int& i, const char** argv, const int argc, unsigned int /*def*/)
{
  return qAddOptionGeneric<unsigned int>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> unsigned int {
      return strtoul(a, nullptr, 0);
    },
    settings.doDefault);
}
template <>
inline int qAddOptionType<long long int>(qConfigSettings<long long int>& settings, long long int& ref, int& i, const char** argv, const int argc, long long int /*def*/)
{
  return qAddOptionGeneric<long long int>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> long long int {
      return strtoll(a, nullptr, 0);
    },
    settings.doDefault);
}
template <>
inline int qAddOptionType<unsigned long long int>(qConfigSettings<unsigned long long int>& settings, unsigned long long int& ref, int& i, const char** argv, const int argc, unsigned long long int /*def*/)
{
  return qAddOptionGeneric<unsigned long long int>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> unsigned long long int {
      return strtoull(a, nullptr, 0);
    },
    settings.doDefault);
}
template <>
inline int qAddOptionType<float>(qConfigSettings<float>& settings, float& ref, int& i, const char** argv, const int argc, float /*def*/)
{
  return qAddOptionGeneric<float>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> float {
      return (float)atof(a);
    },
    settings.doDefault);
}
template <>
inline int qAddOptionType<double>(qConfigSettings<double>& settings, double& ref, int& i, const char** argv, const int argc, double /*def*/)
{
  return qAddOptionGeneric<double>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> double {
      return atof(a);
    },
    settings.doDefault);
}
template <>
inline int qAddOptionType<const char*>(qConfigSettings<const char*>& settings, const char*& ref, int& i, const char** argv, const int argc, const char* /*def*/)
{
  return qAddOptionGeneric<const char*>(
    settings, ref, i, argv, argc, settings.set, [](const char* a) -> const char* {
      return a;
    },
    settings.doDefault);
}

// Checks and messages for additional settings
template <typename T>
static inline int qAddOptionMinMax(qConfigSettings<T>& settings, T& ref, const char* arg)
{
  if (settings.checkMin && ref < settings.min) {
    std::cout << "Invalid setting for " << arg << ": minimum threshold exceeded (" << ref << " < " << settings.min << ")!\n";
    return (qcrMinFailure);
  }
  if (settings.checkMax && ref > settings.max) {
    std::cout << "Invalid setting for " << arg << ": maximum threshold exceeded (" << ref << " > " << settings.max << ")!\n";
    return (qcrMaxFailure);
  }
  return (0);
}
template <>
inline int qAddOptionMinMax<bool>(qConfigSettings<bool>& /*settings*/, bool& /*ref*/, const char* /*arg*/)
{
  return (0);
}

template <typename T>
inline void qAddOptionMessage(qConfigSettings<T>& settings, T& ref)
{
  if (settings.message) {
    printf(settings.message, ref);
    printf("\n");
  }
}
template <>
inline void qAddOptionMessage<bool>(qConfigSettings<bool>& settings, bool& ref)
{
  if (settings.message) {
    printf(settings.message, ref ? "ON" : "OFF");
    printf("\n");
  }
}

static inline void qConfigHelp(const char* subConfig = nullptr, int followSub = 0)
{
  if (followSub < 2) {
    printf("Usage Info:");
  }

#define QCONFIG_HELP
#include "qconfig.h"
#undef QCONFIG_HELP
}

// Create parser for configuration
static inline int qConfigParse(int argc, const char** argv, const char* /*filename*/)
{
  for (int i = 1; i < argc; i++) {
    const char* thisoption = argv[i];
  repeat:
    bool found = false;
#define QCONFIG_PARSE
#include "qconfig.h"
#undef QCONFIG_PARSE
    if (found == false) {
      printf("Invalid argument: %s\n", argv[i]);
      return (1);
      goto repeat; // Suppress GCC warnings, this is never executed, label might be accessed from qconfig.h
    }
  }
  return (0);
}
} // end namespace qConfig

// Main parse function called from outside
int qConfigParse(int argc, const char** argv, const char* filename) { return (qConfig::qConfigParse(argc, argv, filename)); }

// Print current config settings
void qConfigPrint()
{
#define QCONFIG_PRINT
#include "qconfig.h"
#undef QCONFIG_PRINT
}
