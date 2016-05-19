///
/// @file    Foo.cxx
/// @author  Barthelemy von Haller
///

#include "ExampleModule1/Foo.h"

#include <iostream>

namespace AliceO2 {
namespace Examples {
namespace ExampleModule1 {

void Foo::greet()
{
  std::cout << "Hello ExampleModule2 world!!" << std::endl;
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
