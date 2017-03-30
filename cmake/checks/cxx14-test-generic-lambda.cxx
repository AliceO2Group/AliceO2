///
/// \file cxx14-test-generic-lambda.cxx
/// \brief Generic lambdas check
/// \author Adam Wegrzynek <adam.wegrzynek@cern.ch>
///

auto glambda = [] (auto a) { return a; };

int main()
{
  int number = 44;
  return (glambda(number) == number) ? 0 : 1;
}
