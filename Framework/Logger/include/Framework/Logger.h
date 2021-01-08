// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_LOGGER_H_
#define O2_FRAMEWORK_LOGGER_H_

#include <fairlogger/Logger.h>

#define O2DEBUG(...) LOGF(debug, __VA_ARGS__)
#define O2INFO(...) LOGF(info, __VA_ARGS__)
#define O2ERROR(...) LOGF(error, __VA_ARGS__)

#endif // O2_FRAMEWORK_LOGGER_H_
