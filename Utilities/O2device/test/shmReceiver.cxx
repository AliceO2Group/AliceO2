#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <cstring>
#include <zmq.h>
#include <cstdlib> //std::system
#include <cstddef>
#include <cassert>
#include <utility>
#include <chrono>
#include <thread>
#include "../include/O2device/SharedMemory.h"

using namespace AliceO2;
int main(int argc, char* argv[]) {
  namespace bi = boost::interprocess;

  //Construct managed shared memory
  SharedMemory::Manager segman(2000);

  void* zmqContext = zmq_ctx_new();
  //RECEIVER
  void* receiver = zmq_socket(zmqContext, ZMQ_PULL);
  zmq_connect(receiver, "tcp://localhost:2223");
  //zmq_setsockopt(receiver, ZMQ_SUBSCRIBE, "", 0);
  //int timeout = 1000;
  //zmq_setsockopt(receiver, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
  //zmq_setsockopt(receiver, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));

  const size_t messageLen = 100;
  char message[messageLen];
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  int niter = 10000000;
  unsigned int i = 0;
  for  (int i=0; i<niter; ++i) {
    //while(true) {
    //  ++i;
    int rc = zmq_recv(receiver, message, messageLen, 0);
    if (rc==0) break;
    //if (rc>0) message[rc] = '\0';
    //if (rc<0) {printf("rc: %i, err: %s message: %s\n",rc, zmq_strerror(errno), &message[0]);}

    SharedMemory::HandleType handle = *reinterpret_cast<SharedMemory::HandleType*>(&message[0]);
    auto block = segman.getBlock(handle);

    //std::cout << "handle: " << handle << " ptr " << segman.Segment()->get_address_from_handle(handle) << std::endl;
    //printf("data: %s\n", block->data());
  }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
    << "elapsed time: " << elapsed_seconds.count() << "s\n";

  zmq_close(receiver);
  zmq_ctx_term(zmqContext);

  return 0;
  }
