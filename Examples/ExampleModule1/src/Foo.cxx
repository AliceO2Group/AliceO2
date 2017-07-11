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
/// @file    Foo.cxx
/// @author  Barthelemy von Haller
///

#include "ExampleModule1/Foo.h"
#include "Bar.h"

#include <iostream>

namespace o2 {
namespace Examples {
namespace ExampleModule1 {

void Foo::greet()
{
  std::cout << "Hello world from ExampleModule1::Foo" << std::endl;
  Bar bar;
  bar.greet();
}

int Foo::returnsN(int n)
{

  /// \todo This is how you can markup a todo in your code that will show up in the documentation of your project.
  /// \bug This is how you annotate a bug in your code.
  return n;
}

} // namespace ExampleModule1
} // namespace Examples
} // namespace AliceO2
