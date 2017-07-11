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
/// \file cxx14-test-binary-literals.cxx
/// \brief Binary literals check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

int main()
{
  int bin42 = 0b00101010;
  int dec42 = 42;
  return (bin42 == dec42) ? 0 : 1;
}
