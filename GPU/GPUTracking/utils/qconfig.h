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

/// \file qconfig.h
/// \author David Rohr

#ifndef QCONFIG_H_GENERAL
#define QCONFIG_H_GENERAL
extern int qConfigParse(int argc, const char** argv, const char* filename = nullptr);
extern void qConfigPrint();
namespace qConfig
{
enum qConfigRetVal { qcrOK = 0,
                     qcrError = 1,
                     qcrMinFailure = 2,
                     qcrMaxFailure = 3,
                     qcrHelp = 4,
                     qcrCmd = 5,
                     qcrArgMissing = 6,
                     qcrArgIncomplete = 7,
                     qcrArrayOverflow = 8 };
}
#endif

#if !defined(QCONFIG_HEADER_GUARD_NO_INCLUDE) || defined(QCONFIG_GENRTC)

#define AddArrayDefaults(...) \
  {                           \
    __VA_ARGS__               \
  }

#ifdef QCONFIG_PARSE

#define QCONFIG_COMPARE(name, optname, optnameshort)                                                                                                                                           \
  (thisoption[0] == '-' && ((thisoption[1] == '-' && optname[0] != 0 && strcmp(&thisoption[2], optname) == 0) ||                                                                               \
                            (preoptshort == 0 && thisoption[1] == optnameshort && thisoption[2] == 0) || (thisoption[1] == '-' && strlen(preopt) == 0 && strcmp(&thisoption[2], name) == 0) || \
                            (preoptshort != 0 && thisoption[1] == preoptshort && thisoption[2] == optnameshort && thisoption[3] == 0) || (thisoption[1] == '-' && strlen(preopt) && strncmp(&thisoption[2], preopt, strlen(preopt)) == 0 && strcmp(&thisoption[2 + strlen(preopt)], name) == 0)))

#define AddOption(name, type, default, optname, optnameshort, ...)                             \
  else if (QCONFIG_COMPARE(#name, optname, optnameshort))                                      \
  {                                                                                            \
    int retVal = qConfigType<type>::qAddOption(tmp.name, i, argv, argc, default, __VA_ARGS__); \
    if (retVal) {                                                                              \
      return (retVal);                                                                         \
    }                                                                                          \
  }

#define AddOptionSet(name, type, value, optname, optnameshort, ...)                                      \
  else if (QCONFIG_COMPARE(optname, "", optnameshort))                                                   \
  {                                                                                                      \
    int retVal = qConfigType<type>::qAddOption(tmp.name, i, nullptr, 0, value, __VA_ARGS__, set(value)); \
    if (retVal) {                                                                                        \
      return (retVal);                                                                                   \
    }                                                                                                    \
  }

#define AddOptionVec(name, type, optname, optnameshort, ...)                             \
  else if (QCONFIG_COMPARE(#name, optname, optnameshort))                                \
  {                                                                                      \
    int retVal = qConfigType<type>::qAddOptionVec(tmp.name, i, argv, argc, __VA_ARGS__); \
    if (retVal) {                                                                        \
      return (retVal);                                                                   \
    }                                                                                    \
  }

#define AddOptionArray(name, type, count, default, optname, optnameshort, ...)                    \
  else if (QCONFIG_COMPARE(#name, optname, optnameshort))                                         \
  {                                                                                               \
    int retVal = qConfigType<type>::qAddOptionArray(tmp.name, count, i, argv, argc, __VA_ARGS__); \
    if (retVal) {                                                                                 \
      return (retVal);                                                                            \
    }                                                                                             \
  }

#define AddSubConfig(name, instance)

#define BeginConfig(name, instance)       \
  {                                       \
    constexpr const char* preopt = "";    \
    constexpr const char preoptshort = 0; \
    name& tmp = instance;                 \
    bool tmpfound = true;                 \
    if (found) {                          \
    }

#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) \
  {                                                                                     \
    constexpr const char* preopt = preoptname;                                          \
    (void)preopt;                                                                       \
    constexpr const char preoptshort = preoptnameshort;                                 \
    (void)preoptshort;                                                                  \
    name& tmp = parent.instance;                                                        \
    (void)tmp;                                                                          \
    bool tmpfound = true;                                                               \
    if (found) {                                                                        \
    }

#define BeginHiddenConfig(name, instance) \
  {                                       \
    bool tmpfound;                        \
    if (0) {                              \
    }

#define EndConfig()          \
  else { tmpfound = false; } \
  if (tmpfound) {            \
    found = true;            \
  }                          \
  }

#define AddHelp(cmd, cmdshort)                 \
  else if (QCONFIG_COMPARE(cmd, "", cmdshort)) \
  {                                            \
    const char* arg = getArg(i, argv, argc);   \
    qConfigHelp(arg ? arg : preopt);           \
    return (qcrHelp);                          \
  }

#define AddHelpAll(cmd, cmdshort)              \
  else if (QCONFIG_COMPARE(cmd, "", cmdshort)) \
  {                                            \
    const char* arg = getArg(i, argv, argc);   \
    qConfigHelp(arg ? arg : "", true);         \
    return (qcrHelp);                          \
  }

#define AddCommand(cmd, cmdshort, command, help) \
  else if (QCONFIG_COMPARE(cmd, "", cmdshort))   \
  {                                              \
    if (command) {                               \
      return (qcrCmd);                           \
    }                                            \
  }

#define AddShortcut(cmd, cmdshort, forward, help, ...)             \
  else if (QCONFIG_COMPARE(cmd, "", cmdshort))                     \
  {                                                                \
    const char* options[] = {"", __VA_ARGS__, nullptr};            \
    const int nOptions = sizeof(options) / sizeof(options[0]) - 1; \
    qConfigParse(nOptions, options, nullptr);                      \
    thisoption = forward;                                          \
    goto repeat;                                                   \
  }

// End QCONFIG_PARSE
#elif defined(QCONFIG_HELP)
#define AddOption(name, type, default, optname, optnameshort, ...) qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), qon_mxstr(default), optname, optnameshort, preopt, preoptshort, 0, __VA_ARGS__);
#define AddOptionSet(name, type, value, optname, optnameshort, ...) qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), qon_mxstr(value), optname, optnameshort, preopt, preoptshort, 1, __VA_ARGS__);
#define AddOptionVec(name, type, optname, optnameshort, ...) qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type), nullptr, optname, optnameshort, preopt, preoptshort, 2, __VA_ARGS__);
#define AddOptionArray(name, type, count, default, optname, optnameshort, ...) qConfigType<type>::qConfigHelpOption(qon_mxstr(name), qon_mxstr(type) "[" qon_mxstr(count) "]", nullptr, optname, optnameshort, preopt, preoptshort, 2, __VA_ARGS__);
#define AddSubConfig(name, instance)                       \
  printf("\t%s\n\n", qon_mxcat(qConfig_subconfig_, name)); \
  if (followSub) {                                         \
    qConfigHelp(qon_mxstr(name), 2);                       \
  }
#define BeginConfig(name, instance)              \
  if (subConfig == nullptr || *subConfig == 0) { \
    constexpr const char* preopt = "";           \
    constexpr const char preoptshort = 0;        \
    printf("\n");
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...)                                                                                                                   \
  const char* qon_mxcat(qConfig_subconfig_, name) = preoptnameshort == 0 ? (qon_mxstr(name) ": --" preoptname "\n\t\t" descr) : (qon_mxstr(name) ": -" qon_mxstr('a') " (--" preoptname ")\n\t\t" descr); \
  (void)qon_mxcat(qConfig_subconfig_, name);                                                                                                                                                              \
  if (subConfig == nullptr || strcmp(subConfig, followSub == 2 ? qon_mxstr(name) : preoptname) == 0) {                                                                                                    \
    constexpr const char* preopt = preoptname;                                                                                                                                                            \
    (void)preopt;                                                                                                                                                                                         \
    constexpr const char preoptshort = preoptnameshort;                                                                                                                                                   \
    (void)preoptshort;                                                                                                                                                                                    \
    char argBuffer[2] = {preoptnameshort, 0};                                                                                                                                                             \
    printf("\n  %s: (--%s%s%s)\n", descr, preoptname, preoptnameshort == 0 ? "" : " or -", argBuffer);
#define BeginHiddenConfig(name, instance) {
#define EndConfig() }
#define AddHelp(cmd, cmdshort) qConfigType<void*>::qConfigHelpOption("help", "help", nullptr, cmd, cmdshort, preopt, preoptshort, 3, "Show usage information");
#define AddHelpAll(cmd, cmdshort) qConfigType<void*>::qConfigHelpOption("help all", "help all", nullptr, cmd, cmdshort, preopt, preoptshort, 3, "Show usage info including all subparameters");
#define AddCommand(cmd, cmdshort, command, help) qConfigType<void*>::qConfigHelpOption("command", "command", nullptr, cmd, cmdshort, preopt, preoptshort, 4, help);
#define AddShortcut(cmd, cmdshort, forward, help, ...) qConfigType<void*>::qConfigHelpOption("shortcut", "shortcut", nullptr, cmd, cmdshort, preopt, preoptshort, 4, help);
#define AddHelpText(text) printf("\n    " text ":\n");

// End QCONFIG_HELP
#elif defined(QCONFIG_PRINT)
#define AddOption(name, type, default, optname, optnameshort, ...) std::cout << "\t" << blockName << qon_mxstr(name) << ": " << qConfig::print_type(qconfig_tmp_object.name) << "\n";
#define AddVariable(name, type, default) std::cout << "\t" << blockName << qon_mxstr(name) << ": " << qConfig::print_type(qconfig_tmp_object.name) << "\n";
#define AddOptionSet(name, type, value, optname, optnameshort, ...)
#define AddOptionVec(name, type, optname, optnameshort, ...)            \
  {                                                                     \
    std::cout << "\t" << blockName << qon_mxstr(name) << "[]: ";        \
    for (unsigned int i = 0; i < qconfig_tmp_object.name.size(); i++) { \
      if (i) {                                                          \
        std::cout << ", ";                                              \
      }                                                                 \
      std::cout << qConfig::print_type(qconfig_tmp_object.name[i]);     \
    }                                                                   \
    std::cout << "\n";                                                  \
  }
#define AddOptionArray(name, type, count, default, optname, optnameshort, ...)                                                     \
  {                                                                                                                                \
    std::cout << "\t" << blockName << qon_mxstr(name) << "[" << count << "]: " << qConfig::print_type(qconfig_tmp_object.name[0]); \
    for (int i = 1; i < count; i++) {                                                                                              \
      std::cout << ", " << qConfig::print_type(qconfig_tmp_object.name[i]);                                                        \
    }                                                                                                                              \
    std::cout << "\n";                                                                                                             \
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
  void qConfigPrint(const name& qconfig_tmp_object, std::string blockName = "")         \
  {                                                                                     \
    std::cout << "\n\t" << qon_mxstr(name) << ":\n";
#define BeginHiddenConfig(name, instance) BeginSubConfig(name, instance, x, x, x, x)

#define EndConfig() }

// End QCONFIG_PRINT
#elif defined(QCONFIG_INSTANCE)
#define BeginNamespace(name) \
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
#elif defined(QCONFIG_PRINT_RTC)
#define AddOption(name, type, default, optname, optnameshort, help, ...) out << qon_mxstr(type) << " " << qon_mxstr(name) << ";\n";
#define AddVariable(name, type, default) out << qon_mxstr(type) << " " << qon_mxstr(name) << ";\n";
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) out << qon_mxstr(type) << " " << qon_mxstr(name) << "[" << qon_mxstr(count) << "];\n";
#define AddOptionVec(name, type, optname, optnameshort, help, ...) out << "std::vector<" << qon_mxstr(type) << "> " << qon_mxstr(name) << ";\n";
#define AddVariableRTC(name, type, default)                                                                                                                            \
  if (useConstexpr) {                                                                                                                                                  \
    out << "static constexpr " << qon_mxstr(type) << " " << qon_mxstr(name) << " = " << qConfig::print_type(std::get<const qConfigCurrentType*>(tSrc)->name) << ";\n"; \
    out << qon_mxstr(type) << " " << qon_mxstr(qon_mxcat(_dummy_, name)) << ";\n";                                                                                     \
  } else {                                                                                                                                                             \
    AddOption(name, type, default, optname, optnameshort, help);                                                                                                       \
  }
#define AddOptionRTC(name, type, default, optname, optnameshort, help, ...) AddVariableRTC(name, type, default)
#define BeginConfig(name, instance)  \
  {                                  \
    using qConfigCurrentType = name; \
    out << "struct " << qon_mxstr(name) << " {\n";
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) BeginConfig(name, instance)
#define EndConfig() \
  out << "};";      \
  }
#define BeginNamespace(name) out << "namespace " << qon_mxstr(name) << " {\n";
#define EndNamespace() out << "}\n";
#define AddSubConfig(name, instance) out << qon_mxstr(name) << " " << qon_mxstr(instance) << ";";
#define AddOptionSet(...)

// End QCONFIG_PRINT_RTC
#else // Define structures
#if defined(QCONFIG_HEADER_GUARD) && !defined(QCONFIG_GENRTC)
#define QCONFIG_HEADER_GUARD_NO_INCLUDE
#else
#define QCONFIG_HEADER_GUARD

#ifndef BeginNamespace
#define BeginNamespace(name) \
  namespace name             \
  {
#define EndNamespace() }
#endif
#if !defined(QCONFIG_GPU)
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name = default;
#define AddVariable(name, type, default) type name = default;
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) type name[count] = {default};
#define AddOptionVec(name, type, optname, optnameshort, help, ...) std::vector<type> name;
#else
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name;
#define AddVariable(name, type, default) type name;
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) type name[count];
#define AddOptionVec(name, type, optname, optnameshort, help, ...) void* name[sizeof(std::vector<type>) / sizeof(void*)];
#endif
#ifdef QCONFIG_GENRTC
#define AddVariableRTC(name, type, default) \
  static constexpr type name = default;     \
  type _dummy_##name = default;
#define AddOptionRTC(name, type, default, optname, optnameshort, help, ...) AddVariableRTC(name, type, default)
#else
#define AddCustomCPP(...) __VA_ARGS__
#endif
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance) name instance;
#define BeginConfig(name, instance) struct name {
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr, ...) struct name {
#define EndConfig() \
  }                 \
  ;

#endif
#endif

#ifndef QCONFIG_HEADER_GUARD_NO_INCLUDE
#ifndef AddHelp
#define AddHelp(cmd, cmdshort)
#endif
#ifndef AddHelpAll
#define AddHelpAll(cmd, cmdshort)
#endif
#ifndef AddCommand
#define AddCommand(cmd, cmdshort, command)
#endif
#ifndef AddShortcut
#define AddShortcut(cmd, cmdshort, forward, help, ...)
#endif
#ifndef AddVariable
#define AddVariable(name, type, default)
#endif
#ifndef AddHelpText
#define AddHelpText(text)
#endif
#ifndef BeginNamespace
#define BeginNamespace(name)
#endif
#ifndef EndNamespace
#define EndNamespace()
#endif
#ifndef AddCustomCPP
#define AddCustomCPP(...)
#endif
#ifndef AddOptionRTC
#define AddOptionRTC(...) AddOption(__VA_ARGS__)
#endif
#ifndef AddVariableRTC
#define AddVariableRTC(...) AddVariable(__VA_ARGS__)
#endif
#ifndef BeginHiddenConfig
#define BeginHiddenConfig(name, instance) BeginSubConfig(name, instance, , , , )
#endif

#include "qconfigoptions.h"

#undef AddOption
#undef AddOptionRTC
#undef AddVariable
#undef AddVariableRTC
#undef AddOptionSet
#undef AddOptionVec
#undef AddOptionArray
#undef AddArrayDefaults
#undef AddSubConfig
#undef BeginConfig
#undef BeginSubConfig
#undef BeginHiddenConfig
#undef EndConfig
#undef AddHelp
#undef AddHelpAll
#undef AddHelpText
#undef AddCommand
#undef AddShortcut
#undef BeginNamespace
#undef EndNamespace
#undef AddCustomCPP
#ifdef QCONFIG_COMPARE
#undef QCONFIG_COMPARE
#endif

#else
#undef QCONFIG_HEADER_GUARD_NO_INCLUDE
#endif

#endif // QCONFIG_HEADER_GUARD_NO_INCLUDE
