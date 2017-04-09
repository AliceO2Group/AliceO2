using namespace o2::TPC;

bool run_write = true;
bool run_read = true;

std::mutex mtx;

void addData(std::vector<GBTFrame>& data, GBTFrameContainer& container)
{
  unsigned long count = 0;
  unsigned int word0, word1, word2, word3;
  for (std::vector<GBTFrame>::iterator it = data.begin(); it != data.end(); it++){
    it->getGBTFrame(word3,word2,word1,word0);
    container.addGBTFrame(word3,word2,word1,word0);
//    std::cout << *it << std::endl;
    ++count;
  }
  mtx.lock();
  std::cout << "Inserted " << count << " GBT Frames" << std::endl;
  mtx.unlock();
};

void addDataCont(std::vector<GBTFrame>& data, GBTFrameContainer& container)
{
  unsigned long count = 0;
  unsigned int word0, word1, word2, word3;
  while(run_write) {
    for (std::vector<GBTFrame>::iterator it = data.begin(); it != data.end(); it++){
      it->getGBTFrame(word3,word2,word1,word0);
      container.addGBTFrame(word3,word2,word1,word0);
//      container.addGBTFrame(*it);
      ++count;
    }
  }
  mtx.lock();
  std::cout << "Inserted " << count << " GBT Frames" << std::endl;
  mtx.unlock();
};

void readDataCont(GBTFrameContainer& container)
{
  std::vector<HalfSAMPAData> data(5);
  std::vector<HalfSAMPAData> lastData(5);
  unsigned long count = 0;
  while (run_read) {
    std::this_thread::sleep_for(std::chrono::microseconds{10});
    while (container.getData(data)){
      ++count;
      if (data == lastData) std::cout << "Same data twice" << std::endl;
      lastData = data;
    }
  }
  mtx.lock();
  std::cout << "Read " << count << " x 80 (" << count*80 << ") values" << std::endl;
  std::cout << "last data:" << std::endl;
  for (std::vector<HalfSAMPAData>::iterator it = data.begin(); it != data.end(); ++it) {
    std::cout << *it << std::endl;
  }
  std::cout << container.getSize() << " " << container.getNentries() << std::endl;
  mtx.unlock();
};


void test_GBTFrame(std::string infile , int time)
{

  std::cout << infile << std::endl;
//  std::ifstream file(infile);
//
//  int frame_idx;
//  int numGBTXframes = 1000;
//  uint32_t rawData32b;
//  uint16_t gbtx_id;
//  uint32_t gbtx_frame_w3;
//  uint32_t gbtx_frame_w2;
//  uint32_t gbtx_frame_w1;
//  uint32_t gbtx_frame_w0;
//  if (file.is_open()){
//    std::cout << "Reading file : " << infile << std::endl;
//    frame_idx = 0;
//    // read the GBTx Frames
//    while( frame_idx < numGBTXframes && !file.eof() ){
//     file.read( (char*)&rawData32b, sizeof(rawData32b) );
//
//     gbtx_id = (rawData32b >> 16) & 0xFFFF;
//     if( (gbtx_id == 0xdef1) || (gbtx_id == 0xdef4)) {
//       gbtx_frame_w3 = rawData32b;
//       file.read((char*)&gbtx_frame_w2, sizeof(gbtx_frame_w2));
//       file.read((char*)&gbtx_frame_w1, sizeof(gbtx_frame_w1));
//       file.read((char*)&gbtx_frame_w0, sizeof(gbtx_frame_w0));
//     }
//   }
//  }
//  file.close();


  std::ifstream fifofile(infile);
  std::vector<GBTFrame> fifoFrames;
  if (fifofile.is_open()) {
    int word0, word1, word2, word3;
    int c;
    char cc;
    unsigned rawData;
    int gbt_marker;
    std::string str;
                //     counter    :     data
//    while (fifofile >> c       >> cc >> str) {
    while(!fifofile.eof()) {
      fifofile.read((char*)&rawData, sizeof(rawData));
      gbt_marker =  (rawData >> 16) & 0xFFFF;
      if ((gbt_marker == 0xdef1) || (gbt_marker == 0xdef4)) {
        word3 = rawData;
        fifofile.read((char*)&word2, sizeof(word2));
        fifofile.read((char*)&word1, sizeof(word1));
        fifofile.read((char*)&word0, sizeof(word0));
        fifoFrames.emplace_back(word3,word2,word1,word0);
      }
    }
    fifofile.close();

    std::vector<GBTFrame> fifoFrames_withSyncPattern(fifoFrames.begin(), fifoFrames.begin()+ 80);
    std::vector<GBTFrame> fifoFrames_justData(fifoFrames.begin()+80, fifoFrames.end());

    //GBTFrameContainer container(3*10e6,1,1);
    GBTFrameContainer container(1,1);


    container.reset();
    container.setEnableAdcClockWarning(false);
    container.setEnableStoreGBTFrames(true);

    addData(fifoFrames_withSyncPattern,container);
    std::thread t1(addDataCont,std::ref(fifoFrames_justData),std::ref(container));
//    std::thread t2(readDataCont,std::ref(container));

    for (int i = 0; i < time; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
      std::cout << i << " " << container.getNentries() << std::endl;
    }

    run_write = false;
    sleep(2);
//    sleep(2);
    run_read = false;
    t1.join();
//    t2.join();

  }

  GBTFrameContainer container2(1,1);
  container2.setEnableAdcClockWarning(false);
  container2.addGBTFramesFromBinaryFile(infile);
  container2.overwriteAdcClock(-1,0);
  container2.setEnableAdcClockWarning(true);
  container2.reProcessAllFrames();
  if (container2[13103].getAdcClock(1) == 0) 
    container2[13103].setAdcClock(1,0xF);
  else 
    container2[13103].setAdcClock(1,0);
  container2.reProcessAllFrames();
  std::cout << container2.getSize() << " " << container2.getNentries() << std::endl;

  return;
};
