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

#ifndef O2_MCH_DEVIO_DIGITS_PROG_OPTIONS_H
#define O2_MCH_DEVIO_DIGITS_PROG_OPTIONS_H

/* Command line option names that are used by at least two
 * executables (workflow or not).
 */

constexpr const char* OPTNAME_MAX_NOF_TFS = "max-nof-tfs";
constexpr const char* OPTHELP_MAX_NOF_TFS = "max number of timeframes to process";

constexpr const char* OPTNAME_FIRST_TF = "first-tf";
constexpr const char* OPTHELP_FIRST_TF = "first timeframe to process";

constexpr const char* OPTNAME_PRINT_DIGITS = "print-digits";
constexpr const char* OPTHELP_PRINT_DIGITS = "print digits";

constexpr const char* OPTNAME_PRINT_TFS = "print-tfs";
constexpr const char* OPTHELP_PRINT_TFS = "print number of digits and rofs per tf";

#endif
