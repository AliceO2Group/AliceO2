// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifdef __APPLE__
#import <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach/host_info.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#else
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#ifdef __APPLE__
int xnu_write(int pid, void* addr, unsigned char* data, size_t dsize)
{
  assert(dsize != 0);
  assert(addr != nullptr);
  assert(data != nullptr);

  auto* ptxt = (unsigned char*)malloc(dsize);

  assert(ptxt != nullptr);
  memcpy(ptxt, data, dsize);

  mach_port_t task;
  mach_msg_type_number_t dataCunt = dsize;

  kern_return_t kret = task_for_pid(mach_task_self(), pid, &task);
  if (kret != KERN_SUCCESS) {
    printf("task_for_pid failed: %s. Are you root?", mach_error_string(kret));
    return 0;
  }

  vm_protect(task, (vm_address_t)addr, (vm_size_t)dsize, 0,
             VM_PROT_READ | VM_PROT_WRITE | VM_PROT_ALL);

  kret = vm_write(task, (vm_address_t)addr, (pointer_t)ptxt, dataCunt);

  return kret;
}
#elif __linux__
int xnu_write(int pid, void* addr, unsigned char* data, size_t dsize)
{
  if ((ptrace(PTRACE_ATTACH, pid, NULL, NULL)) < 0) {
    perror("ptrace(ATTACH)");
    exit(1);
  }
  int waitStat = 0;
  int waitRes = waitpid(pid, &waitStat, WUNTRACED);
  if (waitRes != pid || !WIFSTOPPED(waitStat)) {
    perror("....:");
    printf("Something went wrong...\n");
    exit(1);
  }

  if ((ptrace(PTRACE_POKEDATA, pid, addr, data))) {
    perror("pokedata");
  }

  if ((ptrace(PTRACE_DETACH, pid, NULL, NULL) < 0)) {
    perror("ptrace(DETACH)");
    exit(1);
  }
  return 0;
}
#endif

// Writes a 4-byte value to the specified address in the target process
// If the address in question points to private_o2_log_<signpost>::stacktrace
// This will have the side effect of enabling the signpost.
int main(int argc, char** argv)
{
  // Use getopt_long to parse command line arguments
  // -p pid
  // -a address
  // -s stacktrace level (default 1, 0 to disable)
  static struct option long_options[] = {
    {"pid", required_argument, nullptr, 'p'},
    {"address", required_argument, nullptr, 'a'},
    {"stacktrace", required_argument, nullptr, 's'},
    {nullptr, 0, nullptr, 0}};
  int opt;
  pid_t pid;
  int stacktrace = 1;
  void* addr;
  while ((opt = getopt_long(argc, argv, "p:a:", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'p':
        pid = atoi(optarg);
        break;
      case 'a':
        addr = (void*)strtoul(optarg, nullptr, 16);
        break;
      case 's':
        stacktrace = strtol(optarg, nullptr, 10);
        break;
      default:
        printf("Usage: %s -p pid -a address [-s level]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
  xnu_write(pid, addr, (unsigned char*)&stacktrace, 4);
  return 0;
}
