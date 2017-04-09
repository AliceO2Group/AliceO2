using namespace o2::TPC;

void test_fixedPoint(Int_t nEvents = 10, std::string mcEngine = "TGeant3")
{
  HwFixedPoint a(-7.6,24,14);
  HwFixedPoint b = 7.6;

  HwFixedPoint e((int)0xAAAAAAAA,5,1);
  HwFixedPoint f((int)0xAAAAAAAA,4,1);
  HwFixedPoint g((int)0xAAAAAAAA,3,1);
  HwFixedPoint h((int)0xAAAAAAAA,2,1);
  HwFixedPoint i((int)0xAAAAAAAA,1,1);

//  b.setFractionPrecision(3);
//  b.setIntegerPrecision(5);

  std::cout << "a:\t\t\t"               << a     << "\t" << (float) a << std::endl;
  std::cout << "b:\t\t\t"               << b     << "\t" << (float) b << std::endl;
  std::cout << std::endl;

  std::cout << "Cast operators:" << std::endl;
  std::cout << "(bool) a:\t\t"          << (bool) a << std::endl;
  std::cout << "(bool) b:\t\t"          << (bool) b << std::endl;
  std::cout << "(int) a:\t\t"           << (int) a << std::endl;
  std::cout << "(int) b:\t\t"           << (int) b << std::endl;
  std::cout << "(unsigned) a:\t\t"      << (unsigned) a << std::endl;
  std::cout << "(unsigned) b:\t\t"      << (unsigned) b << std::endl;
  std::cout << "(float) a:\t\t"         << (float) a << std::endl;
  std::cout << "(float) b:\t\t"         << (float) b << std::endl;
  std::cout << "(double) a:\t\t"        << (double) a << std::endl;
  std::cout << "(double) b:\t\t"        << (double) b << std::endl;
  std::cout << std::endl;

  std::cout << "Assignment operators:" << std::endl;
  a = (int)5;       std::cout << "a = (int) 5:\t\t"       << a << std::endl;
  a = (unsigned)7;  std::cout << "a = (unsigend) 7:\t"    << a << std::endl;
  a = (float)3.2;   std::cout << "a = (float) 3.2:\t"     << a << std::endl;
  a = (double)4.89; std::cout << "a = (double) 4.89:\t"   << a << std::endl;
  HwFixedPoint c(18.123,32,17); 
  a = c;            std::cout << "a = (HwFixedPoint) "    << c << ":\t" << a << std::endl;
  std::cout << std::endl;

  HwFixedPoint d(3.12545,14,3); 
  c = d;
  std::cout << "Addition operators:" << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << "c:\t\t\t"               << c << std::endl;
  std::cout << "a + b:\t\t\t"           << a + b << std::endl;
  std::cout << "b + a:\t\t\t"           << b + a << std::endl;
  std::cout << "a += c:\t\t\t"          << (a += c) << std::endl;
  std::cout << "c += b:\t\t\t"          << (c += b) << std::endl;
  std::cout << "b += b:\t\t\t"          << (b += b) << std::endl;
  std::cout << "a + (int)3:\t\t"        << (a + (int)3) << std::endl;
  std::cout << "b + (float)2.1:\t\t"    << (b + (float)2.1) << std::endl;
  std::cout << "c + (double)4.7:\t"     << (c + (double)4.7) << std::endl;
  std::cout << "a += (int)3:\t\t"       << (a += (int)3) << std::endl;
  std::cout << "b += (float)2.1:\t"     << (b += (float)2.1) << std::endl;
  std::cout << "c += (double)4.7:\t"    << (c += (double)4.7) << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << "a++:\t\t\t"             << a++ << std::endl;
  std::cout << "++b:\t\t\t"             << ++b << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << std::endl;

  std::cout << "Subtraction operators:" << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << "c:\t\t\t"               << c << std::endl;
  std::cout << "a - b:\t\t\t"           << a - b << std::endl;
  std::cout << "b - a:\t\t\t"           << b - a << std::endl;
  std::cout << "a -= c:\t\t\t"          << (a -= c) << std::endl;
  std::cout << "c -= b:\t\t\t"          << (c -= b) << std::endl;
  std::cout << "b -= b:\t\t\t"          << (b -= b) << std::endl;
  std::cout << "a - (int)3:\t\t"        << (a - (int)3) << std::endl;
  std::cout << "b - (float)2.1:\t\t"    << (b - (float)2.1) << std::endl;
  std::cout << "c - (double)4.7:\t"     << (c - (double)4.7) << std::endl;
  std::cout << "a -= (int)3:\t\t"       << (a -= (int)3) << std::endl;
  std::cout << "b -= (float)2.1:\t"     << (b -= (float)2.1) << std::endl;
  std::cout << "c -= (double)4.7:\t"    << (c -= (double)4.7) << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << "a--:\t\t\t"             << a-- << std::endl;
  std::cout << "--b:\t\t\t"             << --b << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << std::endl;

  std::cout << "Multiplication operators:" << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << "c:\t\t\t"               << c << std::endl;
  std::cout << "a * b:\t\t\t"           << a * b << std::endl;
  std::cout << "b * a:\t\t\t"           << b * a << std::endl;
  std::cout << "a *= c:\t\t\t"          << (a *= c) << std::endl;
  std::cout << "c *= b:\t\t\t"          << (c *= b) << std::endl;
  std::cout << "b *= b:\t\t\t"          << (b *= b) << std::endl;
  std::cout << "a * (int)3:\t\t"        << (a * (int)3) << std::endl;
  std::cout << "b * (float)2.1:\t\t"    << (b * (float)2.1) << std::endl;
  std::cout << "c * (double)4.7:\t"     << (c * (double)4.7) << std::endl;
  std::cout << "a *= (int)3:\t\t"       << (a *= (int)3) << std::endl;
  std::cout << "b *= (float)2.1:\t"     << (b *= (float)2.1) << std::endl;
  std::cout << "c *= (double)4.7:\t"    << (c *= (double)4.7) << std::endl;
  std::cout << std::endl;

  std::cout << "Division operators:" << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << "c:\t\t\t"               << c << std::endl;
  std::cout << "a / b:\t\t\t"           << a / b << std::endl;
  std::cout << "b / a:\t\t\t"           << b / a << std::endl;
  std::cout << "a /= c:\t\t\t"          << (a /= c) << std::endl;
  std::cout << "c /= b:\t\t\t"          << (c /= b) << std::endl;
  std::cout << "b /= b:\t\t\t"          << (b /= b) << std::endl;
  std::cout << "a / (int)3:\t\t"        << (a / (int)3) << std::endl;
  std::cout << "b / (float)2.1:\t\t"    << (b / (float)2.1) << std::endl;
  std::cout << "c / (double)4.7:\t"     << (c / (double)4.7) << std::endl;
  std::cout << "a /= (int)3:\t\t"       << (a /= (int)3) << std::endl;
  std::cout << "b /= (float)2.1:\t"     << (b /= (float)2.1) << std::endl;
  std::cout << "c /= (double)4.7:\t"    << (c /= (double)4.7) << std::endl;
  std::cout << std::endl;

  std::cout << "Comparator operators:" << std::endl;
  std::cout << "a:\t\t\t"               << a << std::endl;
  std::cout << "b:\t\t\t"               << b << std::endl;
  std::cout << "a < b:\t\t\t"           << (a < b)  << std::endl;
  std::cout << "a <= b:\t\t\t"          << (a <= b) << std::endl;
  std::cout << "a > b:\t\t\t"           << (a > b)  << std::endl;
  std::cout << "a >= b:\t\t\t"          << (a >= b) << std::endl;
  std::cout << "a == b:\t\t\t"          << (a == b) << std::endl;
  std::cout << "a == a:\t\t\t"          << (a == a) << std::endl;
  std::cout << "a != b:\t\t\t"          << (a != b) << std::endl;
  std::cout << "a != a:\t\t\t"          << (a != a) << std::endl;
  std::cout << std::endl;
  std::cout << "a < 7:\t\t\t"           << (a < 7)  << std::endl;
  std::cout << "a <= 7:\t\t\t"          << (a <= 7) << std::endl;
  std::cout << "a > 7:\t\t\t"           << (a > 7)  << std::endl;
  std::cout << "a >= 7:\t\t\t"          << (a >= 7) << std::endl;
  std::cout << "a == 7:\t\t\t"          << (a == 7) << std::endl;
  std::cout << "a != 7:\t\t\t"          << (a != 7) << std::endl;
  std::cout << std::endl;
  std::cout << "b < 0.3:\t\t"           << (b <  0.3)  << std::endl;
  std::cout << "b <= 0.3:\t\t"          << (b <= 0.3) << std::endl;
  std::cout << "b > 0.3:\t\t"           << (b >  0.3)  << std::endl;
  std::cout << "b >= 0.3:\t\t"          << (b >= 0.3) << std::endl;
  std::cout << "b == 0.3:\t\t"          << (b == 0.3) << std::endl;
  std::cout << "b != 0.3:\t\t"          << (b != 0.3) << std::endl;

  return;
}
