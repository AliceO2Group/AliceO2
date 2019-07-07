#include "Ex4/A.h"
#include <iostream>

ClassImp(ex4::A);

namespace ex4
{
A::A()
{
  std::cout << "Hello from ex4::A ctor\n";
}
int A::value() const
{
  return 42;
}
} // namespace ex4
