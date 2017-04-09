///
/// @file    Bar.cxx
/// @author  Barthelemy von Haller
///

#include "Bar.h"

#include <iostream>

namespace o2 {
namespace Examples {
namespace ExampleModule2 {

Bar::Bar()
= default;

Bar::~Bar()
= default;

void Bar::greet()
{
  std::cout << "Hello world from ExampleModule2::Bar" << std::endl;
}

int Bar::returnsN(int n)
{

  /// \todo This is how you can markup a todo in your code that will show up in the documentation of your project.
  /// \bug This is how you annotate a bug in your code.
  return n;
}

} // namespace ExampleModule2
} // namespace Examples
} // namespace AliceO2
