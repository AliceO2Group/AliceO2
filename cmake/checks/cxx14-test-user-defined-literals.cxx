///
/// \file cxx14-test-user-defined-literals.cxx
/// \brief Standard user-defined literals check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

#include <string>
#include <chrono>
#include <complex>

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

bool testComplex() {
  using namespace std::literals::complex_literals;
  auto zl = 1i; 
  std::complex<double> z(0, 1);
  return (zl == z); 
}

int main()
{
  return (testComplex() && testString() && testChrono()) ? 0 : 1;
}
