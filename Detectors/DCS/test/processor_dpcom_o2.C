using namespace o2::dcs;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

int processor_dpcom_o2()
{

  std::unordered_map<DPID, DPVAL> dpmap;

  o2::dcs::DCSProcessor dcsproc;
  std::vector<DPID> aliasChars;
  std::vector<DPID> aliasInts;
  std::vector<DPID> aliasDoubles;
  std::vector<DPID> aliasUInts;
  std::vector<DPID> aliasBools;
  std::vector<DPID> aliasStrings;
  std::vector<DPID> aliasTimes;
  std::vector<DPID> aliasBinaries;

  DeliveryType typechar = RAW_CHAR;
  std::string dpAliaschar = "TestChar0";
  DPID charVar(dpAliaschar, typechar);
  aliasChars.push_back(charVar);

  DeliveryType typeint = RAW_INT;
  std::string dpAliasint0 = "TestInt0";
  DPID intVar0(dpAliasint0, typeint);
  aliasInts.push_back(intVar0);

  std::string dpAliasint1 = "TestInt1";
  DPID intVar1(dpAliasint1, typeint);
  aliasInts.push_back(intVar1);

  std::string dpAliasint2 = "TestInt2";
  DPID intVar2(dpAliasint2, typeint);
  aliasInts.push_back(intVar2);

  DeliveryType typedouble = RAW_DOUBLE;
  std::string dpAliasdouble0 = "TestDouble0";
  DPID doubleVar0(dpAliasdouble0, typedouble);
  aliasDoubles.push_back(doubleVar0);

  std::string dpAliasdouble1 = "TestDouble1";
  DPID doubleVar1(dpAliasdouble1, typedouble);
  aliasDoubles.push_back(doubleVar1);

  std::string dpAliasdouble2 = "TestDouble2";
  DPID doubleVar2(dpAliasdouble2, typedouble);
  aliasDoubles.push_back(doubleVar2);

  std::string dpAliasdouble3 = "TestDouble3";
  DPID doubleVar3(dpAliasdouble3, typedouble);
  aliasDoubles.push_back(doubleVar3);

  DeliveryType typestring = RAW_STRING;
  std::string dpAliasstring0 = "TestString0";
  DPID stringVar0(dpAliasstring0, typestring);
  aliasStrings.push_back(stringVar0);

  dcsproc.init(aliasChars, aliasInts, aliasDoubles, aliasUInts, aliasBools, aliasStrings, aliasTimes, aliasBinaries);

  uint16_t flags = 0;
  uint16_t milliseconds = 0;
  TDatime currentTime;
  uint32_t seconds = currentTime.Get();
  uint64_t* payload = new uint64_t[7];

  // loop that emulates the number of times the DCS DataPoints are sent
  for (auto k = 0; k < 4; k++) {
    payload[0] = (uint64_t)k + 33; // adding 33 to have visible chars and strings

    DPVAL valchar(flags, milliseconds + k * 10, seconds + k, payload, typechar);
    DPVAL valint(flags, milliseconds + k * 10, seconds + k, payload, typeint);
    DPVAL valdouble(flags, milliseconds + k * 10, seconds + k, payload, typedouble);
    DPVAL valstring(flags, milliseconds + k * 10, seconds + k, payload, typestring);

    dpmap[charVar] = valchar;
    dpmap[intVar0] = valint;
    dpmap[intVar1] = valint;
    dpmap[intVar2] = valint;
    dpmap[doubleVar0] = valdouble;
    dpmap[doubleVar1] = valdouble;
    dpmap[doubleVar2] = valdouble;
    dpmap[stringVar0] = valstring;
    if (k != 3)
      dpmap[doubleVar3] = valdouble; // to test the case when a DP is not updated
    std::cout << "index = " << k << std::endl;
    std::cout << charVar << std::endl
              << valchar << " --> " << (char)valchar.payload_pt1 << std::endl;
    std::cout << intVar0 << std::endl
              << valint << " --> " << (int)valchar.payload_pt1 << std::endl;
    std::cout << intVar1 << std::endl
              << valint << " --> " << (int)valchar.payload_pt1 << std::endl;
    std::cout << intVar2 << std::endl
              << valint << " --> " << (int)valchar.payload_pt1 << std::endl;
    std::cout << doubleVar0 << std::endl
              << valdouble << " --> " << (double)valchar.payload_pt1 << std::endl;
    std::cout << doubleVar1 << std::endl
              << valdouble << " --> " << (double)valchar.payload_pt1 << std::endl;
    std::cout << doubleVar2 << std::endl
              << valdouble << " --> " << (double)valchar.payload_pt1 << std::endl;
    std::cout << doubleVar3 << std::endl
              << valdouble << " --> " << (double)valchar.payload_pt1 << std::endl;
    char tt[56];
    memcpy(&tt[0], &valstring.payload_pt1, 56);
    printf("tt = %s\n", tt);
    std::cout << stringVar0 << std::endl
              << valstring << " --> " << tt << std::endl;

    dcsproc.process(dpmap);
  }
  std::cout << "The map has " << dpmap.size() << " entries" << std::endl;
  return 0;
}
