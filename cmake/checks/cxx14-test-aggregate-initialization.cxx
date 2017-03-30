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
  S test{ 1, 2, 3, 4, 5, 6 };
  return 0;
}
