// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfig.h
/// \author David Rohr

#include <vector>

#define AddArrayDefaults(...) \
  {                           \
    __VA_ARGS__               \
  }

#ifdef QCONFIG_PARSE

#define QCONFIG_COMPARE(optname, optnameshort)                                                                                                                                                    \
  (thisoption[0] == '-' && ((preoptshort == 0 && thisoption[1] == optnameshort && thisoption[2] == 0) || (thisoption[1] == '-' && strlen(preopt) == 0 && strcmp(&thisoption[2], optname) == 0) || \
                            (preoptshort != 0 && thisoption[1] == preoptshort && thisoption[2] == optnameshort && thisoption[3] == 0) || (thisoption[1] == '-' && strlen(preopt) && strncmp(&thisoption[2], preopt, strlen(preopt)) == 0 && strcmp(&thisoption[2 + strlen(preopt)], optname) == 0)))

#define AddOption(name, type, default, optname, optnameshort, ...)                             \
  else if (QCONFIG_COMPARE(optname, optnameshort))                                             \
  {                                                                                            \
    int retVal = qConfigType<type>::qAddOption(tmp.name, i, argv, argc, default, __VA_ARGS__); \
    if (retVal) {                                                                              \
      return (retVal);                                                                         \
    }                                                                                          \
  }

#define AddOptionSet(name, type, value, optname, optnameshort, ...)                                      \
  else if (QCONFIG_COMPARE(optname, optnameshort))                                                       \
  {                                                                                                      \
    int retVal = qConfigType<type>::qAddOption(tmp.name, i, nullptr, 0, value, __VA_ARGS__, set(value)); \
    if (retVal) {                                                                                        \
      return (retVal);                                                                                   \
    }                                                                                                    \
  }

#define AddOptionVec(name, type, optname, optnameshort, ...)                             \
  else if (QCONFIG_COMPARE(optname, optnameshort))                                       \
  {                                                                                      \
    int retVal = qConfigType<type>::qAddOptionVec(tmp.name, i, argv, argc, __VA_ARGS__); \
    if (retVal) {                                                                        \
      return (retVal);                                                                   \
    }                                                                                    \
  }

#define AddOptionArray(name, type, count, default, optname, optnameshort, ...)                    \
  else if (QCONFIG_COMPARE(optname, optnameshort))                                                \
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
      ;                                   \
    }

#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
  {                                                                                \
    constexpr const char* preopt = preoptname;                                     \
    constexpr const char preoptshort = preoptnameshort;                            \
    name& tmp = parent.instance;                                                   \
    bool tmpfound = true;                                                          \
    if (found) {                                                                   \
      ;                                                                            \
    }

#define EndConfig()          \
  else { tmpfound = false; } \
  if (tmpfound) {            \
    found = true;            \
  }                          \
  }

#define AddHelp(cmd, cmdshort)               \
  else if (QCONFIG_COMPARE(cmd, cmdshort))   \
  {                                          \
    const char* arg = getArg(i, argv, argc); \
    qConfigHelp(arg ? arg : preopt);         \
    return (qcrHelp);                        \
  }

#define AddHelpAll(cmd, cmdshort)            \
  else if (QCONFIG_COMPARE(cmd, cmdshort))   \
  {                                          \
    const char* arg = getArg(i, argv, argc); \
    qConfigHelp(arg ? arg : "", true);       \
    return (qcrHelp);                        \
  }

#define AddCommand(cmd, cmdshort, command, help) \
  else if (QCONFIG_COMPARE(cmd, cmdshort))       \
  {                                              \
    if (command) {                               \
      return (qcrCmd);                           \
    }                                            \
  }

#define AddShortcut(cmd, cmdshort, forward, help, ...)             \
  else if (QCONFIG_COMPARE(cmd, cmdshort))                         \
  {                                                                \
    const char* options[] = { "", __VA_ARGS__, nullptr };          \
    const int nOptions = sizeof(options) / sizeof(options[0]) - 1; \
    qConfigParse(nOptions, options, nullptr);                      \
    thisoption = forward;                                          \
    goto repeat;                                                   \
  }

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
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr)                                                                                                                        \
  const char* qon_mxcat(qConfig_subconfig_, name) = preoptnameshort == 0 ? (qon_mxstr(name) ": --" preoptname "\n\t\t" descr) : (qon_mxstr(name) ": -" qon_mxstr('a') " (--" preoptname ")\n\t\t" descr); \
  if (subConfig == nullptr || strcmp(subConfig, followSub == 2 ? qon_mxstr(name) : preoptname) == 0) {                                                                                                    \
    constexpr const char* preopt = preoptname;                                                                                                                                                            \
    constexpr const char preoptshort = preoptnameshort;                                                                                                                                                   \
    char argBuffer[2] = { preoptnameshort, 0 };                                                                                                                                                           \
    printf("\n  %s: (--%s%s%s)\n", descr, preoptname, preoptnameshort == 0 ? "" : " or -", argBuffer);
#define EndConfig() }
#define AddHelp(cmd, cmdshort) qConfigType<void*>::qConfigHelpOption("help", "help", nullptr, cmd, cmdshort, preopt, preoptshort, 3, "Show usage information");
#define AddHelpAll(cmd, cmdshort) qConfigType<void*>::qConfigHelpOption("help all", "help all", nullptr, cmd, cmdshort, preopt, preoptshort, 3, "Show usage info including all subparameters");
#define AddCommand(cmd, cmdshort, command, help) qConfigType<void*>::qConfigHelpOption("command", "command", nullptr, cmd, cmdshort, preopt, preoptshort, 4, help);
#define AddShortcut(cmd, cmdshort, forward, help, ...) qConfigType<void*>::qConfigHelpOption("shortcut", "shortcut", nullptr, cmd, cmdshort, preopt, preoptshort, 4, help);
#define AddHelpText(text) printf("\n    " text ":\n");

#elif defined(QCONFIG_PRINT)
#define AddOption(name, type, default, optname, optnameshort, ...) std::cout << "\t" << qon_mxstr(name) << ": " << tmp.name << "\n";
#define AddVariable(name, type, default) std::cout << "\t" << qon_mxstr(name) << ": " << tmp.name << "\n";
#define AddOptionSet(name, type, value, optname, optnameshort, ...)
#define AddOptionVec(name, type, optname, optnameshort, ...) \
  {                                                          \
    std::cout << "\t" << qon_mxstr(name) << "[]: ";          \
    for (unsigned int i = 0; i < tmp.name.size(); i++) {     \
      if (i) {                                               \
        std::cout << ", ";                                   \
      } /*std::cout << tmp.name[i];*/                        \
    }                                                        \
    std::cout << "\n";                                       \
  }
#define AddOptionArray(name, type, count, default, optname, optnameshort, ...)    \
  {                                                                               \
    std::cout << "\t" << qon_mxstr(name) << "[" << count << "]: " << tmp.name[0]; \
    for (int i = 1; i < count; i++) {                                             \
      std::cout << ", " << tmp.name[i];                                           \
    }                                                                             \
    std::cout << "\n";                                                            \
  }
#define AddSubConfig(name, instance)
#define AddHelpText(text) printf("    " text ":\n");
#define BeginConfig(name, instance) \
  {                                 \
    name& tmp = instance;
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
  {                                                                                \
    name& tmp = parent.instance;
#define EndConfig() }

#elif defined(QCONFIG_INSTANCE)
#define AddOption(name, type, default, optname, optnameshort, help, ...) name(default),
#define AddVariable(name, type, default) name(default),
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...) name(),
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) name default,
#define AddSubConfig(name, instance) instance(),
#define BeginConfig(name, instance) \
  name instance;                    \
  name::name() :
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) name::name() :
#define EndConfig() \
  _qConfigDummy() {}

#elif defined(QCONFIG_EXTERNS)
#define AddOption(name, type, default, optname, optnameshort, help, ...)
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddOptionVec(name, type, optname, optnameshort, help, ...)
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance)
#define BeginConfig(name, instance) extern "C" name instance;
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr)
#define EndConfig()
#undef QCONFIG_EXTERNS
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

#else
#ifdef QCONFIG_HEADER_GUARD
#define QCONFIG_HEADER_GUARD_NO_INCLUDE
#else
#define QCONFIG_HEADER_GUARD

#if defined(QCONFIG_CPP11_INIT) && !defined(QCONFIG_GPU)
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name = default;
#define AddVariable(name, type, default) type name = default;
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance) name instance;
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) type name[count] = { default };
#define AddOptionVec(name, type, optname, optnameshort, help, ...) std::vector<type> name;
#define BeginConfig(name, instance) \
  struct name {
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
  struct name {
#define EndConfig() \
  }                 \
  ;
#else
#define AddOption(name, type, default, optname, optnameshort, help, ...) type name;
#define AddVariable(name, type, default) type name;
#define AddOptionSet(name, type, value, optname, optnameshort, help, ...)
#define AddSubConfig(name, instance) name instance;
#define AddOptionArray(name, type, count, default, optname, optnameshort, help, ...) type name[count];
#define EndConfig()           \
  qConfigDummy _qConfigDummy; \
  }                           \
  ;
#ifdef QCONFIG_GPU
#define AddOptionVec(name, type, optname, optnameshort, help, ...) void* name[sizeof(std::vector<type>) / sizeof(void*)];
#define BeginConfig(name, instance) \
  struct name {
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
  struct name {
struct qConfigDummy {
};
#else
#define AddOptionVec(name, type, optname, optnameshort, help, ...) std::vector<type> name;
#define BeginConfig(name, instance) \
  struct name {                     \
    name();                         \
    name(const name& s);            \
    name& operator=(const name& s);
#define BeginSubConfig(name, instance, parent, preoptname, preoptnameshort, descr) \
  struct name {                                                                    \
    name();                                                                        \
    name(const name& s);                                                           \
    name& operator=(const name& s);
;
struct qConfigDummy {
  qConfigDummy() {}
};
#define QCONFIG_EXTERNS
#endif
#endif
#endif
#endif

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

#ifndef QCONFIG_HEADER_GUARD_NO_INCLUDE
#include "qconfigoptions.h"
#endif

#undef AddOption
#undef AddVariable
#undef AddOptionSet
#undef AddOptionVec
#undef AddOptionArray
#undef AddArrayDefaults
#undef AddSubConfig
#undef BeginConfig
#undef BeginSubConfig
#undef EndConfig
#undef AddHelp
#undef AddHelpAll
#undef AddHelpText
#undef AddCommand
#undef AddShortcut
#ifdef QCONFIG_COMPARE
#undef QCONFIG_COMPARE
#endif

#ifdef QCONFIG_EXTERNS
#include "qconfig.h"
#endif
