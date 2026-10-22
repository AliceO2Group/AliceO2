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

/// \file qconfig_wrapinclude.h
/// \author David Rohr

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
#ifndef AddOptionArrayRTC
#define AddOptionArrayRTC(...) AddOptionArray(__VA_ARGS__)
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
#undef AddOptionArrayRTC
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
