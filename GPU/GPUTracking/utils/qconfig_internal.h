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

/// \file qconfig_internal.h
/// \author David Rohr

// For internal use by qconfig.cxx
// External usage is possible, but not fully supported

#include "qconfig_general.h"
#include "qconfig_helpers.h"

#ifdef QCONFIG_CHECK // qcheck(expression) as additional setting of an option: expression is evaluated by qConfigCheck if the option has not the default value (qconfig_check.inc), can use val (value of the option) and get<T>() (object of type T passed to qConfigCheck)
#define qcheck(...) \
  ::qConfig::qcheck_result { static_cast<bool>(__VA_ARGS__), #__VA_ARGS__ }
#else
#define qcheck(...) \
  ::qConfig::qcheck_ignore {}
#endif

#ifdef QCONFIG_PARSE // Included in the constructor of qConfigOptionRegistry: registers the names and the handler of each option

#define AddOption(name, type, default, optname, optnameshort, ...) \
  add(preopt, preoptshort, #name, optname, optnameshort, qConfigType<type>::qNeedsArg(__VA_ARGS__), nullptr, [ref = &tmp.name](int32_t& i, const char** argv, const int argc) -> int32_t { return qConfigType<type>::qAddOption(*ref, i, argv, argc, default, __VA_ARGS__); });

#define AddOptionSet(name, type, value, optname, optnameshort, ...) \
  add(preopt, preoptshort, optname, "", optnameshort, 0, nullptr, [ref = &tmp.name](int32_t& i, const char**, const int) -> int32_t { return qConfigType<type>::qAddOption(*ref, i, nullptr, 0, value, __VA_ARGS__, set(value)); });

#define AddOptionVec(name, type, optname, optnameshort, ...) \
  add(preopt, preoptshort, #name, optname, optnameshort, qConfigType<type>::qNeedsArg(__VA_ARGS__), nullptr, [ref = &tmp.name](int32_t& i, const char** argv, const int argc) -> int32_t { return qConfigType<type>::qAddOptionVec(*ref, i, argv, argc, __VA_ARGS__); });

#define AddOptionArray(name, type, count, default, optname, optnameshort, ...) \
  add(preopt, preoptshort, #name, optname, optnameshort, qConfigType<type>::qNeedsArg(__VA_ARGS__), nullptr, [ref = &tmp.name](int32_t& i, const char** argv, const int argc) -> int32_t { return qConfigType<type>::qAddOptionArray(*ref, count, i, argv, argc, __VA_ARGS__); });

#define AddSubConfig(name, instance)

#define BeginConfig(name, instance)                        \
  {                                                        \
    [[maybe_unused]] constexpr const char* preopt = "";    \
    [[maybe_unused]] constexpr const char preoptshort = 0; \
    [[maybe_unused]] name& tmp = instance;

#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) \
  {                                                                                     \
    [[maybe_unused]] constexpr const char* preopt = preoptname;                         \
    [[maybe_unused]] constexpr const char preoptshort = preoptnameshort;                \
    [[maybe_unused]] name& tmp = parent.instance;

#define BeginHiddenConfig(name, instance) {

#define EndConfig() }

#define AddHelp(cmd, cmdshort)                                                                                                              \
  add(preopt, preoptshort, cmd, "", cmdshort, 2, nullptr, [helpConfig = preopt](int32_t& i, const char** argv, const int argc) -> int32_t { \
    const char* arg = getArg(i, argv, argc);                                                                                                \
    qConfigHelp(arg ? arg : helpConfig);                                                                                                    \
    return (qcrHelp);                                                                                                                       \
  });

#define AddHelpAll(cmd, cmdshort)                                                                                        \
  add(preopt, preoptshort, cmd, "", cmdshort, 2, nullptr, [](int32_t& i, const char** argv, const int argc) -> int32_t { \
    const char* arg = getArg(i, argv, argc);                                                                             \
    qConfigHelp(arg ? arg : "", true);                                                                                   \
    return (qcrHelp);                                                                                                    \
  });

#define AddCommand(cmd, cmdshort, command, help) \
  add(preopt, preoptshort, cmd, "", cmdshort, 0, nullptr, [](int32_t&, const char**, const int) -> int32_t { return (command) ? qcrCmd : 0; });

#define AddShortcut(cmd, cmdshort, forward, help, ...) /* Parses the listed options, then the parser continues with the forward option, which takes the arguments */ \
  add(preopt, preoptshort, cmd, "", cmdshort, 0, forward, [](int32_t&, const char**, const int) -> int32_t {                                                         \
    const char* options[] = {"", __VA_ARGS__, nullptr};                                                                                                              \
    const int32_t nOptions = sizeof(options) / sizeof(options[0]) - 1;                                                                                               \
    qConfigParse(nOptions, options, nullptr);                                                                                                                        \
    return 0;                                                                                                                                                        \
  });

// End QCONFIG_PARSE
#elif defined(QCONFIG_HELP)
#define AddOption(name, type, default, optname, optnameshort, ...) qconfig_help_options.emplace_back(qon_mxstr(name), [=]() { qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), qon_mxstr(default), optname, optnameshort, preopt, preoptshort, 0, __VA_ARGS__); });
#define AddOptionSet(name, type, value, optname, optnameshort, ...) qconfig_help_options.emplace_back(qon_mxstr(name), [=]() { qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), qon_mxstr(value), optname, optnameshort, preopt, preoptshort, 1, __VA_ARGS__); });
#define AddOptionVec(name, type, optname, optnameshort, ...) qconfig_help_options.emplace_back(qon_mxstr(name), [=]() { qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), nullptr, optname, optnameshort, preopt, preoptshort, 2, __VA_ARGS__); });
#define AddOptionArray(name, type, count, default, optname, optnameshort, ...) qconfig_help_options.emplace_back(qon_mxstr(name), [=]() { qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type) "[" qon_mxstr(count) "]", nullptr, optname, optnameshort, preopt, preoptshort, 2, __VA_ARGS__); });
#define AddSubConfig(name, instance)                         \
  qconfig_help_other.emplace_back(nullptr, [=]() {           \
    printf("\t%s\n\n", qon_mxcat(qConfig_subconfig_, name)); \
    if (followSub) {                                         \
      qConfigHelp(qon_mxstr(name), 2);                       \
    }                                                        \
  });
#define BeginConfig(name, instance)              \
  if (subConfig == nullptr || *subConfig == 0) { \
    constexpr const char* preopt = "";           \
    constexpr const char preoptshort = 0;        \
    printf("\n");                                \
    qConfigHelpEntries qconfig_help_options, qconfig_help_other;
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...)                                                                                                                   \
  const char* qon_mxcat(qConfig_subconfig_, name) = preoptnameshort == 0 ? (qon_mxstr(name) ": --" preoptname "\n\t\t" descr) : (qon_mxstr(name) ": -" qon_mxstr('a') " (--" preoptname ")\n\t\t" descr); \
  (void)qon_mxcat(qConfig_subconfig_, name);                                                                                                                                                              \
  if (subConfig == nullptr || strcmp(subConfig, followSub == 2 ? qon_mxstr(name) : preoptname) == 0) {                                                                                                    \
    [[maybe_unused]] constexpr const char* preopt = preoptname;                                                                                                                                           \
    [[maybe_unused]] constexpr const char preoptshort = preoptnameshort;                                                                                                                                  \
    char argBuffer[2] = {preoptnameshort, 0};                                                                                                                                                             \
    printf("\n  %s: (--%s%s%s)\n", descr, preoptname, preoptnameshort == 0 ? "" : " or -", argBuffer);                                                                                                    \
    qConfigHelpEntries qconfig_help_options, qconfig_help_other;
#define BeginHiddenConfig(name, instance) \
  {                                       \
    qConfigHelpEntries qconfig_help_options, qconfig_help_other;
#define EndConfig()                                                 \
  qConfigHelpPrintSorted(qconfig_help_options, qconfig_help_other); \
  }
#define AddHelp(cmd, cmdshort) qconfig_help_other.emplace_back(nullptr, [=]() { qConfigType<void*>::qConfigHelpOption("help", "help", nullptr, cmd, cmdshort, preopt, preoptshort, 3, "Show usage information"); });
#define AddHelpAll(cmd, cmdshort) qconfig_help_other.emplace_back(nullptr, [=]() { qConfigType<void*>::qConfigHelpOption("help all", "help all", nullptr, cmd, cmdshort, preopt, preoptshort, 3, "Show usage info including all subparameters"); });
#define AddCommand(cmd, cmdshort, command, help) qconfig_help_other.emplace_back(nullptr, [=]() { qConfigType<void*>::qConfigHelpOption("command", "command", nullptr, cmd, cmdshort, preopt, preoptshort, 4, help); });
#define AddShortcut(cmd, cmdshort, forward, help, ...) qconfig_help_other.emplace_back(nullptr, [=]() { qConfigType<void*>::qConfigHelpOption("shortcut", "shortcut", nullptr, cmd, cmdshort, preopt, preoptshort, 4, help); });
#define AddHelpText(text) qconfig_help_other.emplace_back(nullptr, []() { printf("\n    " text ":\n"); });

// End QCONFIG_HELP
#elif defined(QCONFIG_PRINT)
template <class T>
void qConfigPrint(const T& qconfig_tmp_object, const std::string& blockName); // (Re)declare in the namespace of the include for the specializations below, default argument only in qconfig_general.h
#define AddOption(name, type, default, optname, optnameshort, ...) std::cout << "\t" << blockName << qon_mxstr(name) << ": " << ::qConfig::print_type(qconfig_tmp_object.name) << "\n";
#define AddVariable(name, type, default) std::cout << "\t" << blockName << qon_mxstr(name) << ": " << ::qConfig::print_type(qconfig_tmp_object.name) << "\n";
#define AddOptionSet(name, type, value, optname, optnameshort, ...)
#define AddOptionVec(name, type, optname, optnameshort, ...)          \
  {                                                                   \
    std::cout << "\t" << blockName << qon_mxstr(name) << "[]: ";      \
    for (uint32_t i = 0; i < qconfig_tmp_object.name.size(); i++) {   \
      if (i) {                                                        \
        std::cout << ", ";                                            \
      }                                                               \
      std::cout << ::qConfig::print_type(qconfig_tmp_object.name[i]); \
    }                                                                 \
    std::cout << "\n";                                                \
  }
#define AddOptionArray(name, type, count, default, optname, optnameshort, ...)                                                       \
  {                                                                                                                                  \
    std::cout << "\t" << blockName << qon_mxstr(name) << "[" << count << "]: " << ::qConfig::print_type(qconfig_tmp_object.name[0]); \
    for (int32_t i = 1; i < count; i++) {                                                                                            \
      std::cout << ", " << ::qConfig::print_type(qconfig_tmp_object.name[i]);                                                        \
    }                                                                                                                                \
    std::cout << "\n";                                                                                                               \
  }
#define AddSubConfig(name, instance) qConfigPrint(qconfig_tmp_object.instance, blockName + qon_mxstr(instance.));
#define AddHelpText(text) printf("    " text ":\n");
#define BeginConfig(name, instance)                      \
  void qConfigPrint(const name& qconfig_tmp_object);     \
  bool qon_mxcat(qprint_global_, instance) = []() { qprint_global.emplace_back([]() { qConfigPrint(instance); }); return true; }(); \
  void qConfigPrint(const name& qconfig_tmp_object)      \
  {                                                      \
    std::string blockName = "";
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) \
  template <>                                                                           \
  void qConfigPrint<name>(const name& qconfig_tmp_object, const std::string& blockName) \
  {                                                                                     \
    std::cout << "\n\t" << qon_mxstr(name) << ":\n";
#define BeginHiddenConfig(name, instance) BeginSubConfig(name, instance, x, x, x, x)

#define EndConfig() }

// End QCONFIG_PRINT
#elif defined(QCONFIG_INSTANCE)
#define BeginNamespace(name) \
  using namespace name;      \
  namespace name             \
  {
#define EndNamespace() }
#define AddOption(name, type, default, optname, optnameshort, help, ...)
#define AddVariable(name, type, default)
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance)
#define BeginConfig(name, instance) name instance;
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...)
#define EndConfig()

// End QCONFIG_INSTANCE
#elif defined(QCONFIG_PRINT_RTC_ARRAYS) // GPUconstexpr() arrays at namespace scope for the AddOptionArrayRTC entries, printed before the structs by qConfigPrintRtc if gpuConstexprArrayInConstantMem is set
#define AddOption(name, type, default, optname, optnameshort, help, ...)
#define AddVariable(name, type, default)
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddOptionSet(...)
#define AddSubConfig(name, instance)
#define AddVariableRTC(name, type, default)
#define AddOptionRTC(name, type, default, optname, optnameshort, help, ...)
#define AddOptionArrayRTC(name, type, count, default, optname, optnameshort, help, ...)                                                     \
  out << "GPUconstexpr() " << qon_mxstr(type) << " qConfigRTC_" << qConfigCurrentName << "_" << qon_mxstr(name) << "[" << count << "] = {"; \
  for (int32_t i = 0; i < count; i++) {                                                                                                     \
    out << (i ? ", " : "") << ::qConfig::print_type(std::get<const qConfigCurrentType*>(tSrc)->name[i], true);                              \
  }                                                                                                                                         \
  out << "};\n";
#define BeginConfig(name, instance)  \
  {                                  \
    using qConfigCurrentType = name; \
    [[maybe_unused]] constexpr const char* qConfigCurrentName = qon_mxstr(name);
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) BeginConfig(name, instance)
#define EndConfig() }
#define BeginNamespace(name) out << "namespace " << qon_mxstr(name) << " {\n";
#define EndNamespace() out << "}\n";

// End QCONFIG_PRINT_RTC_ARRAYS
#elif defined(QCONFIG_PRINT_RTC)
#define AddOption(name, type, default, optname, optnameshort, help, ...) out << qon_mxstr(type) << " " << qon_mxstr(name) << ";\n";
#define AddVariable(name, type, default) out << qon_mxstr(type) << " " << qon_mxstr(name) << ";\n";
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) out << qon_mxstr(type) << " " << qon_mxstr(name) << "[" << qon_mxstr(count) << "];\n";
#define AddOptionVec(name, type, optname, optnameshort, help, ...) out << "std::vector<" << qon_mxstr(type) << "> " << qon_mxstr(name) << ";\n";
#define AddVariableRTC(name, type, default)                                                                                                                                    \
  if (useConstexpr) {                                                                                                                                                          \
    out << "static constexpr " << qon_mxstr(type) << " " << qon_mxstr(name) << " = " << ::qConfig::print_type(std::get<const qConfigCurrentType*>(tSrc)->name, true) << ";\n"; \
    out << qon_mxstr(type) << " " << qon_mxstr(qon_mxcat(_dummy_, name)) << ";\n";                                                                                             \
  } else {                                                                                                                                                                     \
    AddOption(name, type, default, optname, optnameshort, help);                                                                                                               \
  }
#define AddOptionRTC(name, type, default, optname, optnameshort, help, ...) AddVariableRTC(name, type, default)
#define AddOptionArrayRTC(name, type, count, default, optname, optnameshort, help, ...)                                                                                                                                              \
  if (useConstexpr && gpuConstexprArrayInConstantMem) { /* CUDA cannot access static constexpr member arrays in device code, so the values are in a GPUconstexpr() array (QCONFIG_PRINT_RTC_ARRAYS), the storage keeps the layout */ \
    const std::string arrayName = std::string("qConfigRTC_") + qConfigCurrentName + "_" + qon_mxstr(name);                                                                                                                           \
    out << "struct " << qon_mxstr(qon_mxcat(qConfigArrayRTC_, name)) << " {\n"                                                                                                                                                       \
        << qon_mxstr(type) << " values[" << count << "];\n";                                                                                                                                                                         \
    out << "template <class I> GPUdi() constexpr " << qon_mxstr(type) << " operator[](I i) const { return " << arrayName << "[i]; }\n";                                                                                              \
    out << "GPUdi() constexpr operator const " << qon_mxstr(type) << "*() const { return " << arrayName << "; }\n} " << qon_mxstr(name) << ";\n";                                                                                    \
  } else if (useConstexpr) {                                                                                                                                                                                                         \
    out << "static constexpr " << qon_mxstr(type) << " " << qon_mxstr(name) << "[" << count << "] = {";                                                                                                                              \
    for (int32_t i = 0; i < count; i++) {                                                                                                                                                                                            \
      out << (i ? ", " : "") << ::qConfig::print_type(std::get<const qConfigCurrentType*>(tSrc)->name[i], true);                                                                                                                     \
    }                                                                                                                                                                                                                                \
    out << "};\n"                                                                                                                                                                                                                    \
        << qon_mxstr(type) << " " << qon_mxstr(qon_mxcat(_dummy_, name)) << "[" << count << "];\n";                                                                                                                                  \
  } else {                                                                                                                                                                                                                           \
    AddOptionArray(name, type, count, default, optname, optnameshort, help);                                                                                                                                                         \
  }
#define BeginConfig(name, instance)                                              \
  {                                                                              \
    using qConfigCurrentType = name;                                             \
    [[maybe_unused]] constexpr const char* qConfigCurrentName = qon_mxstr(name); \
    out << "struct " << qon_mxstr(name) << " {\n";
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) BeginConfig(name, instance)
#define EndConfig() \
  out << "};\n";    \
  }
#define BeginNamespace(name) out << "namespace " << qon_mxstr(name) << " {\n";
#define EndNamespace() out << "}\n";
#define AddSubConfig(name, instance) out << qon_mxstr(name) << " " << qon_mxstr(instance) << ";";
#define AddOptionSet(...)

// End QCONFIG_PRINT_RTC
#elif defined(QCONFIG_GEN_RTTI)
#define QCONFIG_RTTI(vName, vType, rtc, variable, vector, count, defaultValue, ...)                                                                                                                                                                                                                                                        \
  qConfigRTTIEntry<vType>(retVal,                                                                                                                                                                                                                                                                                                          \
                          T{::qConfig::qConfigRTTIentry{.name = qon_mxstr(vName), .type = qon_mxstr(vType), .isRTC = rtc, .isVariable = variable, .isVector = vector, .offset = static_cast<size_t>(reinterpret_cast<const char*>(&qConfigObject.vName) - reinterpret_cast<const char*>(&qConfigObject)), .configStruct = qConfigStruct}}, \
                          count, *qConfigStructIndex, defaultValue, __VA_ARGS__);
#define AddOption(name, type, default, optname, optnameshort, ...) QCONFIG_RTTI(name, type, false, false, false, 0, print_type(static_cast<type>(default)), __VA_ARGS__)
#define AddOptionRTC(name, type, default, optname, optnameshort, ...) QCONFIG_RTTI(name, type, true, false, false, 0, print_type(static_cast<type>(default)), __VA_ARGS__)
#define AddVariable(name, type, default) QCONFIG_RTTI(name, type, false, true, false, 0, print_type(static_cast<type>(default)), "")
#define AddVariableRTC(name, type, default) QCONFIG_RTTI(name, type, true, true, false, 0, print_type(static_cast<type>(default)), "")
#define AddOptionVec(name, type, optname, optnameshort, ...) QCONFIG_RTTI(name, type, false, false, true, 0, "", __VA_ARGS__)
#define AddOptionArray(name, type, count, default, optname, optnameshort, ...) QCONFIG_RTTI(name, type, false, false, false, count, qon_mxstr(default), __VA_ARGS__)
#define AddOptionArrayRTC(name, type, count, default, optname, optnameshort, ...) QCONFIG_RTTI(name, type, true, false, false, count, qon_mxstr(default), __VA_ARGS__)
#define AddOptionSet(...)
#define AddSubConfig(name, instance)
#define BeginConfig(name, instance)                                                                                                        \
  if constexpr (qConfigRTTISelected<name, Structs...>) {                                                                                   \
    if (const auto qConfigStructIndex = qConfigRTTIStructIndex<name, Structs...>(retStructs, qon_mxstr(name), configStructs != nullptr)) { \
      [[maybe_unused]] constexpr const char* qConfigStruct = qon_mxstr(name);                                                              \
      [[maybe_unused]] const name qConfigObject{}; // Default-constructed instance to compute the member offsets
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) BeginConfig(name, instance)
#define EndConfig() \
  }                 \
  }

// End QCONFIG_GEN_RTTI
#elif defined(QCONFIG_CHECK) // Included in qConfigChecker::check(), blocks of config structs not passed to qConfigCheck are discarded
#define AddOption(name, type, default, optname, optnameshort, ...)                                       \
  {                                                                                                      \
    [[maybe_unused]] const auto& val = qConfigObj.name;                                                  \
    retVal += qConfigCheckOption(qConfigStruct, qon_mxstr(name), val, qConfigDefault.name, __VA_ARGS__); \
  }
#define AddOptionVec(name, type, optname, optnameshort, ...) AddOption(name, type, , optname, optnameshort, __VA_ARGS__)
#define AddOptionArray(name, type, count, default, optname, optnameshort, ...) AddOption(name, type, , optname, optnameshort, __VA_ARGS__)
#define AddOptionSet(...)
#define AddSubConfig(name, instance)
#define BeginConfig(name, instance)                        \
  if constexpr ((std::is_same_v<name, Args> || ...)) {     \
    [[maybe_unused]] const name& qConfigObj = get<name>(); \
    [[maybe_unused]] const name qConfigDefault{};          \
    [[maybe_unused]] constexpr const char* qConfigStruct = qon_mxstr(name);
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) BeginConfig(name, instance)
#define EndConfig() }

// End QCONFIG_CHECK
#else // Define structures

#error Internal error, qconfig_internal called with action type

#endif

#define QCONFIG_INTERNAL
#include "qconfig_wrapinclude.h"
#undef QCONFIG_INTERNAL
#undef qcheck
