using namespace AliceO2::TPC;

void test_GBTFrame()
{
  // Initialize logger
//  FairLogger *logger = FairLogger::GetLogger();
//  logger->SetLogVerbosityLevel("HIGH");
//  logger->SetLogScreenLevel("DEBUG");


  GBTFrame a;
  GBTFrame b(11,25,37,41);

  std::cout << a << std::endl;
  std::cout << b << std::endl;

  std::cout << (unsigned)b.getAdcClock(0) << std::endl;
  std::cout << (unsigned)b.getAdcClock(1) << std::endl;
  std::cout << (unsigned)b.getAdcClock(2) << std::endl;
  std::cout << (unsigned)b.getAdcClock(3) << std::endl;
  std::cout << (unsigned)b.getAdcClock(4) << std::endl;
  std::cout << (unsigned)b.getAdcClock(5) << std::endl;

  std::cout << std::endl << std::endl;
  std::vector<GBTFrame> frames;
  frames.push_back(GBTFrame(0xDEF10F00,0x000F0F0F,0xF0F0F0F0,0x0F0F0000));
  frames.push_back(GBTFrame(0xDEF10F00,0x000F0F0F,0xF0F0F0F0,0x0F0F0000));
  frames.push_back(GBTFrame(0xDEF10F00,0x000F0F0F,0xF0F0F0F0,0x0F0F0000));
  frames.push_back(GBTFrame(0xDEF10F07,0x070F0F0F,0xF0F0F0F0,0x1E7F0707));
  frames.push_back(GBTFrame(0xDEF103C3,0xC30F0F0F,0xF0F0F03C,0x3C33C3C3));
  frames.push_back(GBTFrame(0xDEF103C3,0xC30F0F0F,0xF0F0F03C,0x3C33C3C3));
  frames.push_back(GBTFrame(0xDEF103C3,0xC30F0F0F,0xF0F0F03C,0x3C33C3C3));
  frames.push_back(GBTFrame(0xDEF103C3,0xC30F0F0F,0xF0F0F03C,0x3C33C3C3));
  frames.push_back(GBTFrame(0xDEF10C3C,0x3C0F0F0F,0xF0F0F0C3,0xC3CC3C3C));
  frames.push_back(GBTFrame(0xDEF103C3,0xC30F0F0F,0xF0F0F03C,0x3C33C3C3));
  frames.push_back(GBTFrame(0xDEF10C3C,0x3C0F0F0F,0xF0F0F0C3,0xC3CC3C3C));
  frames.push_back(GBTFrame(0xDEF101E2,0xF20F0F0F,0xF0F0F01C,0x2F23C2D2));
  frames.push_back(GBTFrame(0xDEF1070A,0x6B0F0F0F,0xF0F0F052,0xA7A780D0));
  frames.push_back(GBTFrame(0xDEF10F82,0xF00F0F0F,0xF0F0F07A,0x2F8F8850));
  frames.push_back(GBTFrame(0xDEF107AA,0x600F0F0F,0xF0F0F0D2,0xA72588D2));
  frames.push_back(GBTFrame(0xDEF10D00,0x590F0F0F,0xF0F0F0D2,0x2DA780F8));

  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); it++) {
    int s00 = ((int)it->getHalfWord(0,1,0) << 5) + (int)it->getHalfWord(0,0,0);
    int s01 = ((int)it->getHalfWord(0,3,0) << 5) + (int)it->getHalfWord(0,2,0);
    int s02 = ((int)it->getHalfWord(0,1,1) << 5) + (int)it->getHalfWord(0,0,1);
    int s03 = ((int)it->getHalfWord(0,3,1) << 5) + (int)it->getHalfWord(0,2,1);
    int s10 = ((int)it->getHalfWord(1,1,0) << 5) + (int)it->getHalfWord(1,0,0);
    int s11 = ((int)it->getHalfWord(1,3,0) << 5) + (int)it->getHalfWord(1,2,0);
    int s12 = ((int)it->getHalfWord(1,1,1) << 5) + (int)it->getHalfWord(1,0,1);
    int s13 = ((int)it->getHalfWord(1,3,1) << 5) + (int)it->getHalfWord(1,2,1);
    int s20 = ((int)it->getHalfWord(2,1,0) << 5) + (int)it->getHalfWord(2,0,0);
    int s21 = ((int)it->getHalfWord(2,3,0) << 5) + (int)it->getHalfWord(2,2,0);
    std::cout << std::hex
      << "[ " << "0x" << std::setfill('0') << std::setw(3) << s00 << " " << "0x" << std::setfill('0') << std::setw(3) << s01 << " ][ " << "0x" << std::setfill('0') << std::setw(3) << s02 << " " << "0x" << std::setfill('0') << std::setw(3) << s03 << " ] | " 
      << "[ " << "0x" << std::setfill('0') << std::setw(3) << s10 << " " << "0x" << std::setfill('0') << std::setw(3) << s11 << " ][ " << "0x" << std::setfill('0') << std::setw(3) << s12 << " " << "0x" << std::setfill('0') << std::setw(3) << s13 << " ] | " 
      << "[ " << "0x" << std::setfill('0') << std::setw(3) << s20 << " " << "0x" << std::setfill('0') << std::setw(3) << s21 << " ]"
      << std::dec << std::endl;
  }

  std::cout << std::endl << std::endl;

  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); it++) {
    std::cout << std::hex << "[ " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,0,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,1,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,2,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,3,0)) << " ][ " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,0,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,1,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,2,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,3,1)) << " ] | " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,0,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,1,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,2,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,3,0)) << " ][ " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,0,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,1,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,2,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,3,1)) << " ] | " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,0,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,1,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,2,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,3,0)) << " ]" 
      << std::dec << std::endl;
  }

//  std::cout << std::endl << std::endl << "ADC clock SAMPA0:" << std::endl;
//  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); it++) {
//    std::cout << (int) it->getAdcClock(0) << std::endl;
//  }
//
//  std::cout << std::endl << std::endl << "ADC clock SAMPA1:" << std::endl;
//  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); it++) {
//    std::cout << (int) it->getAdcClock(1) << std::endl;
//  }
//
//  std::cout << std::endl << std::endl << "ADC clock SAMPA2:" << std::endl;
//  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); it++) {
//    std::cout << (int) it->getAdcClock(2) << std::endl;
//  }

  std::cout << std::endl << std::endl;

  std::vector<GBTFrame> frames2;
  std::vector<GBTFrame> frames3;
  unsigned word3, word2, word1, word0;
  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); it++) {
    frames2.push_back(GBTFrame(
          it->getHalfWord(0,0,0),it->getHalfWord(0,1,0),it->getHalfWord(0,2,0),it->getHalfWord(0,3,0),
          it->getHalfWord(0,0,1),it->getHalfWord(0,1,1),it->getHalfWord(0,2,1),it->getHalfWord(0,3,1),
          it->getHalfWord(1,0,0),it->getHalfWord(1,1,0),it->getHalfWord(1,2,0),it->getHalfWord(1,3,0),
          it->getHalfWord(1,0,1),it->getHalfWord(1,1,1),it->getHalfWord(1,2,1),it->getHalfWord(1,3,1),
          it->getHalfWord(2,0),  it->getHalfWord(2,1),  it->getHalfWord(2,2),  it->getHalfWord(2,3),
          it->getAdcClock(0),    it->getAdcClock(1),    it->getAdcClock(2), 0xDEF1
          ));
    std::cout << frames2.back() << std::endl;
    frames2.back().getGBTFrame(word3, word2, word1, word0);
    frames3.push_back(GBTFrame(word3,word2,word1,word0));
  }

  std::cout << std::endl << std::endl;

  for (std::vector<GBTFrame>::iterator it = frames3.begin(); it != frames3.end(); it++) {
    std::cout << std::hex << "[ " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,0,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,1,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,2,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,3,0)) << " ][ " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,0,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,1,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,2,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(0,3,1)) << " ] | " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,0,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,1,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,2,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,3,0)) << " ][ " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,0,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,1,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,2,1)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(1,3,1)) << " ] | " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,0,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,1,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,2,0)) << " " 
      << "0x" << std::setfill('0') << std::setw(2) << ((int)it->getHalfWord(2,3,0)) << " ]" 
      << std::dec << std::endl;
  }

  std::string file = "/misc/alidata120/alice_u/sklewin/alice/fifo_data_0";
  std::cout << file << std::endl;

  ifstream fifofile(file);
  int c;
  char cc;
  std::string str;
  std::vector<GBTFrame> fifoFrames;
              //  counter :  data
  while (fifofile >> c >> cc >> str) {
    sscanf(str.substr( 2,8).c_str(), "%x", &word3);
    sscanf(str.substr(10,8).c_str(), "%x", &word2);
    sscanf(str.substr(18,8).c_str(), "%x", &word1);
    sscanf(str.substr(26,8).c_str(), "%x", &word0);
    fifoFrames.emplace_back(word3,word2,word1,word0);
  }


//  for (std::vector<GBTFrame>::iterator it = fifoFrames.begin(); it != fifoFrames.end(); it++){
//    int s00 = ((int)it->getHalfWord(0,1,0) << 5) + (int)it->getHalfWord(0,0,0);
//    int s01 = ((int)it->getHalfWord(0,3,0) << 5) + (int)it->getHalfWord(0,2,0);
//    int s02 = ((int)it->getHalfWord(0,1,1) << 5) + (int)it->getHalfWord(0,0,1);
//    int s03 = ((int)it->getHalfWord(0,3,1) << 5) + (int)it->getHalfWord(0,2,1);
//    int s10 = ((int)it->getHalfWord(1,1,0) << 5) + (int)it->getHalfWord(1,0,0);
//    int s11 = ((int)it->getHalfWord(1,3,0) << 5) + (int)it->getHalfWord(1,2,0);
//    int s12 = ((int)it->getHalfWord(1,1,1) << 5) + (int)it->getHalfWord(1,0,1);
//    int s13 = ((int)it->getHalfWord(1,3,1) << 5) + (int)it->getHalfWord(1,2,1);
//    int s20 = ((int)it->getHalfWord(2,1,0) << 5) + (int)it->getHalfWord(2,0,0);
//    int s21 = ((int)it->getHalfWord(2,3,0) << 5) + (int)it->getHalfWord(2,2,0);
//    std::cout << *it << " : " << std::hex
//      << "[ " << "0x" << std::setfill('0') << std::setw(3) << s00 << " " << "0x" << std::setfill('0') << std::setw(3) << s01 << " ][ " << "0x" << std::setfill('0') << std::setw(3) << s02 << " " << "0x" << std::setfill('0') << std::setw(3) << s03 << " ] | " 
//      << "[ " << "0x" << std::setfill('0') << std::setw(3) << s10 << " " << "0x" << std::setfill('0') << std::setw(3) << s11 << " ][ " << "0x" << std::setfill('0') << std::setw(3) << s12 << " " << "0x" << std::setfill('0') << std::setw(3) << s13 << " ] | " 
//      << "[ " << "0x" << std::setfill('0') << std::setw(3) << s20 << " " << "0x" << std::setfill('0') << std::setw(3) << s21 << " ]"
//      << std::dec << std::endl;
//
//  }


  TStopwatch timer;


  timer.Start();
  GBTFrameContainer container(5000,1,0);
  container.setEnableAdcClockWarning(false);
//  container.setEnableSyncPatternWarning(false);
  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); ++it) {
    container.addGBTFrame(
          it->getHalfWord(0,0,0),it->getHalfWord(0,1,0),it->getHalfWord(0,2,0),it->getHalfWord(0,3,0),
          it->getHalfWord(0,0,1),it->getHalfWord(0,1,1),it->getHalfWord(0,2,1),it->getHalfWord(0,3,1),
          it->getHalfWord(1,0,0),it->getHalfWord(1,1,0),it->getHalfWord(1,2,0),it->getHalfWord(1,3,0),
          it->getHalfWord(1,0,1),it->getHalfWord(1,1,1),it->getHalfWord(1,2,1),it->getHalfWord(1,3,1),
          it->getHalfWord(2,0),  it->getHalfWord(2,1),  it->getHalfWord(2,2),  it->getHalfWord(2,3),
          it->getAdcClock(0),    it->getAdcClock(1),    it->getAdcClock(2), 0xDEF1
        );
  }

//  for (int i = 0; i < 1000; i++) {
//    container.addGBTFrame(
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0x7, 0x3, 0x1, 0xDEF1
//        );
//    for (int j = 0; j < 3; j++) {
//      container.addGBTFrame(
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0xF, 0xF, 0xF, 0xDEF1
//          );
//    }
//    container.addGBTFrame(
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0, 0, 0, 0,
//        0x8, 0xC, 0xE, 0xDEF1
//        );
//    for (int j = 0; j < 3; j++) {
//      container.addGBTFrame(
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0x0, 0x0, 0x0, 0xDEF1
//          );
//    }
//
//    if (i == 7) {
//      container.addGBTFrame(
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0, 0, 0, 0,
//          0x8, 0xC, 0xE, 0xDEF1
//          );
//    }
//  }
  timer.Stop();

//  std::vector<Digit> digits;
//  while (container.getDigits(&digits)) {
//    std::cout << digits.size() << std::endl;
//  }

  std::cout << std::endl << std::endl;
  std::cout << std::endl << std::endl;

  FairSystemInfo sysInfo;
  Float_t maxMemory=sysInfo.GetMaxMemory();
  std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
  std::cout << maxMemory;
  std::cout << "</DartMeasurement>" << std::endl;

  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();

  Float_t cpuUsage=ctime/rtime;
  std::cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
  std::cout << cpuUsage;
  std::cout << "</DartMeasurement>" << std::endl;
  std::cout << rtime << std::endl;

  std::cout << container.getSize() << " " << container.getNentries() << std::endl;
  

  container.reset();
  container.setEnableAdcClockWarning(false);

//  std::ofstream ofile;
//  ofile.open("./out_3.txt");
  for (std::vector<GBTFrame>::iterator it = fifoFrames.begin(); it != fifoFrames.end(); it++){
//  for (std::vector<GBTFrame>::iterator it = frames.begin(); it != frames.end(); it++){
    std::cout << *it << std::endl;
//    ofile << *it << std::endl;
    container.addGBTFrame(*it);
  }
//  ofile.close();

  std::vector<std::vector<SAMPAData>> allData;
  std::vector<SAMPAData> data(5);
  while (container.getData(&data)) {
    allData.push_back(data);
    data.clear();
  }

  for (auto &d : allData[0])
    std::cout << d << std::endl;

  std::cout << allData.size() << std::endl;

//
//
////  std::vector<Digit> digits2;
////  std::cout << container.getDigits(&digits2) << std::endl;
//
//  for (std::vector<GBTFrame>::iterator it = container.begin(); it != container.end(); ++it) {
//    std::cout << *it << std::endl;
//  }
//  std::cout << std::endl;
//
//  container[14] = container[2];
//  for (std::vector<GBTFrame>::iterator it = container.begin(); it != container.end(); ++it) {
//    std::cout << *it << std::endl;
//  }
//
//  container.reProcessAllFrames();

  return;
}
