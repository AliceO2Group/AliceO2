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
