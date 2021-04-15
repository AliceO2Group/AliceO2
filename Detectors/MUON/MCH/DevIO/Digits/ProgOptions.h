// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma once

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
