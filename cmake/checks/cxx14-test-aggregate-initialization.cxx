// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file cxx14-test-aggregate-initialization.cxx
/// \brief Aggregate member initialization check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

struct S {
  int x;
  struct Foo {
    int i;
    int j;
    int a[3];
  } b;
};

int main()
{
  S test{1, 2, 3, 4, 5, 6};
  return 0;
}
