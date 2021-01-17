// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_ALIAS_EXPANDER_H
#define O2_DCS_ALIAS_EXPANDER_H

#include <vector>
#include <string>

namespace o2::dcs
{
/**
  * expandAlias converts a single pattern into a list of strings.
  *
  * @param pattern a pattern is made of a number of "XX[YY]" blocks (at least one)
  *
  * where :
  * - XX is any text
  * - YY describes either a integral range or a textual list
  *
  * An integral range is [a..b] where the formatting of the biggest of the 
  * two integers a and b dictates, by default, the formatting of the output 
  * alias. For instance [0..3] is expanded to the set 0,1,2,3 while [00..03] 
  * is expanded to 00,01,02,03. If you want more control on the formatting,
  * you can use a python/fmt format {} e.g. [0..15{:d}] would yields 0,1,
  * 2,...,14,15 simply (no 0 filling).
  *
  * A textual list is simply a list of values separated by commas, 
  * e.g. "vMon,iMon"
  *
  * @returns a vector of strings containing all the possible expansions of
  * the pattern. That vector is not guaranteed to be sorted.
  *
  * For example, pattern=DET[A,B]/Channel[000,002]/[iMon,vMon] yields : 
  *
  * - DETA/Channel000/iMon
  * - DETA/Channel001/iMon
  * - DETA/Channel002/iMon
  * - DETA/Channel000/vMon
  * - DETA/Channel001/vMon
  * - DETA/Channel002/vMon
  * - DETB/Channel000/iMon
  * - DETB/Channel001/iMon
  * - DETB/Channel002/iMon
  * - DETB/Channel000/vMon
  * - DETB/Channel001/vMon
  * - DETB/Channel002/vMon

*/
std::vector<std::string> expandAlias(const std::string& pattern);

/** expandAliases converts a list of patterns into a list of strings.
  *
  * each input pattern is treated by expandAlias()
  *
  * @returns a _sorted_ vector of strings containing all the possible
  * expansions of the pattern.
  */
std::vector<std::string> expandAliases(const std::vector<std::string>& patternedAliases);
} // namespace o2::dcs

#endif
