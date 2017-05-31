// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file cxx14-test-user-defined-literals.cxx
/// \brief Standard user-defined literals check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

#include <chrono>
#include <complex>
#include <string>

bool testString()
{
  using namespace std::literals;
  auto strl = "hello world"s;
  std::string str = "hello world";
  if (str.compare(strl) == 0) {
    return true;
  } else {
    return false;
  }
}

bool testChrono()
{
  using namespace std::chrono_literals;
  auto durl = 60s;
  std::chrono::seconds dur(60);
  return (durl == dur);
}

bool testComplex()
{
  using namespace std::literals::complex_literals;
  auto zl = 1i;
  std::complex<double> z(0, 1);
  return (zl == z);
}

int main() { return (testComplex() && testString() && testChrono()) ? 0 : 1; }
