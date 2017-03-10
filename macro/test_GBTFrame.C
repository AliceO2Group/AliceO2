using namespace AliceO2::TPC;

bool run_write = true;
bool run_read = true;

std::mutex mtx;

void addData(std::vector<GBTFrame>& data, GBTFrameContainer& container)
{
  unsigned long count = 0;
  for (std::vector<GBTFrame>::iterator it = data.begin(); it != data.end(); it++){
    container.addGBTFrame(*it);
    std::cout << *it << std::endl;
    ++count;
  }
  mtx.lock();
  std::cout << "Added " << count << " GBT Frames" << std::endl;
  mtx.unlock();
};

void addDataCont(std::vector<GBTFrame>& data, GBTFrameContainer& container)
{
  unsigned long count = 0;
  while(run_write) {
    for (std::vector<GBTFrame>::iterator it = data.begin(); it != data.end(); it++){
      container.addGBTFrame(*it);
      ++count;
    }
  }
  mtx.lock();
  std::cout << "Added " << count << " GBT Frames" << std::endl;
  mtx.unlock();
};

void readDataCont(GBTFrameContainer& container)
{
  std::vector<SAMPAData> data(5);
  unsigned long count = 0;
  while (run_read) {
      std::this_thread::sleep_for(std::chrono::microseconds{10});
    while (container.getData(&data)){
      ++count;
    }// else {
    //}
  }
  mtx.lock();
  std::cout << "Read " << count << " x 5 x 16 (" << count*5*16 << ") values" << std::endl;
  std::cout << "last data:" << std::endl;
  for (std::vector<SAMPAData>::iterator it = data.begin(); it != data.end(); ++it) {
    std::cout << *it << std::endl;
  }
  std::cout << container.getSize() << std::endl;
  mtx.unlock();
};


void test_GBTFrame(std::string infile = "/misc/alidata120/alice_u/sklewin/alice/fifo_data_0")
{

  std::cout << infile << std::endl;

  std::ifstream fifofile(infile);
  int word0, word1, word2, word3;
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

  std::vector<GBTFrame> fifoFrames_withSyncPattern(fifoFrames.begin(), fifoFrames.begin()+ 80);
  std::vector<GBTFrame> fifoFrames_justData(fifoFrames.begin()+80, fifoFrames.end());

  GBTFrameContainer container(5000,0,1);
  GBTFrameContainer container2(5000,1,1);
  container.setEnableAdcClockWarning(false);
  container2.setEnableAdcClockWarning(false);


  container.reset();
  container.setEnableAdcClockWarning(false);
  container2.reset();
  container2.setEnableAdcClockWarning(false);

  addData(fifoFrames_withSyncPattern,container);
  std::thread t1(addDataCont,std::ref(fifoFrames_justData),std::ref(container));
//  std::thread t2(readDataCont,std::ref(container));

  sleep(1);
  run_write = false;
  sleep(2);
  run_read = false;
  t1.join();
//  t2.join();

//  int counter = 0;
//  for (std::vector<GBTFrame>::iterator it = fifoFrames.begin(); it != fifoFrames.end(); it++){
//    std::cout << counter++ << ": " << *it << std::endl;
//    container.addGBTFrame(*it);
//    container2.addGBTFrame(*it);
//  }
//
//  std::vector<std::vector<SAMPAData>> allData;
//  std::vector<SAMPAData> data(5);
//  std::vector<Digit> digits;
//  container.getDigits(&digits,false);
//  while (container.getData(&data) && container2.getData(&data)) {
//    allData.push_back(data);
//    data.clear();
//    container.getDigits(&digits,false);
//  }
//
//  for (auto &d : allData[0])
//    std::cout << d << std::endl;
//
//  std::cout << digits.size() << std::endl;
//  std::cout << allData.size() << std::endl;
//
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
};
