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

// A C++ way to list / iterate files with glob searches

#ifndef O2_FILEITERATOR_H
#define O2_FILEITERATOR_H

#include <vector>
#include <string>
#include <unistd.h>

namespace o2::utils
{

// Return a vector of file names (not directories), with full path information, in a given directory "dir".
// If searchpattern is empty, all files will be returned. Otherwise
// searchpattern will be treated/parsed as a proper regular expression.
std::vector<std::string> listFiles(std::string const& dir, std::string const& searchpattern);

// same in the current dir
std::vector<std::string> listFiles(std::string const& searchpattern);

// create path if absent, account for eventual concurrent creation
void createDirectoriesIfAbsent(std::string const& path);

// A function to expand a string containing shell variables
// to a string in which these vars have been substituted.
// Motivation:: filesystem::exists() does not do this by default
// and I couldn't find information on this. Potentially there is an
// existing solution.
std::string expandShellVarsInFileName(std::string const& input);

} // namespace o2::utils

#endif //O2_FILEITERATOR_H
