///
/// \file cxx14-test-make_unique.cxx
/// \brief std::make_unique check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

#include <memory>

bool checkPointer(int number)
{
  auto pointer = std::make_unique<int>(number);
  return *pointer == number;
}

int main()
{
  bool ret = checkPointer(41);
  return ret ? 0 : 1;
}
