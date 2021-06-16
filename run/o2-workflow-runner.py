#!/usr/bin/env python3

# started February 2021, sandro.wenzel@cern.ch

import re
import subprocess
import shlex
import time
import json
import logging
import os
import signal
import socket
import sys
import traceback
try:
    from graphviz import Digraph
    havegraphviz=True
except ImportError:
    havegraphviz=False

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

sys.setrecursionlimit(100000)

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# first file logger
actionlogger = setup_logger('pipeline_action_logger', 'pipeline_action.log', level=logging.DEBUG)

# second file logger
metriclogger = setup_logger('pipeline_metric_logger', 'pipeline_metric.log')

# for debugging without terminal access
# TODO: integrate into standard logger
def send_webhook(hook, t):
    if hook!=None:
      command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" " + str(t) + "\"}' " + str(hook) + " &> /dev/null"
      os.system(command)

# A fallback solution to getting all child procs
# in case psutil has problems (PermissionError).
# It returns the same list as psutil.children(recursive=True).
def getChildProcs(basepid):
  cmd='''
  childprocs() {
  local parent=$1
  if [ ! "$2" ]; then
    child_pid_list=""
  fi
  if [ "$parent" ] ; then
    child_pid_list="$child_pid_list $parent"
    for childpid in $(pgrep -P ${parent}); do
      childprocs $childpid "nottoplevel"
    done;
  fi
  # return via a string list (only if toplevel)
  if [ ! "$2" ]; then
    echo "${child_pid_list}"
  fi
  }
  '''
  cmd = cmd + '\n' + 'childprocs ' + str(basepid)
  output = subprocess.check_output(cmd, shell=True)
  plist = []
  for p in output.strip().split():
     try:
         proc=psutil.Process(int(p))
     except psutil.NoSuchProcess:
         continue

     plist.append(proc)
  return plist

#
# Code section to find all topological orderings
# of a DAG. This is used to know when we can schedule
# things in parallel.
# Taken from https://www.geeksforgeeks.org/all-topological-sorts-of-a-directed-acyclic-graph/

# class to represent a graph object
class Graph:

    # Constructor
    def __init__(self, edges, N):

        # A List of Lists to represent an adjacency list
        self.adjList = [[] for _ in range(N)]

        # stores in-degree of a vertex
        # initialize in-degree of each vertex by 0
        self.indegree = [0] * N

        # add edges to the undirected graph
        for (src, dest) in edges:

            # add an edge from source to destination
            self.adjList[src].append(dest)

            # increment in-degree of destination vertex by 1
            self.indegree[dest] = self.indegree[dest] + 1


# Recursive function to find all topological orderings of a given DAG
def findAllTopologicalOrders(graph, path, discovered, N, allpaths, maxnumber=1):
    if len(allpaths) >= maxnumber:
        return

    # do for every vertex
    for v in range(N):

        # proceed only if in-degree of current node is 0 and
        # current node is not processed yet
        if graph.indegree[v] == 0 and not discovered[v]:

            # for every adjacent vertex u of v, reduce in-degree of u by 1
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] - 1

            # include current node in the path and mark it as discovered
            path.append(v)
            discovered[v] = True

            # recur
            findAllTopologicalOrders(graph, path, discovered, N, allpaths)

            # backtrack: reset in-degree information for the current node
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] + 1

            # backtrack: remove current node from the path and
            # mark it as undiscovered
            path.pop()
            discovered[v] = False

    # record valid ordering
    if len(path) == N:
        allpaths.append(path.copy())


# get all topological orderings of a given DAG as a list
def printAllTopologicalOrders(graph, maxnumber=1):
    # get number of nodes in the graph
    N = len(graph.adjList)

    # create an auxiliary space to keep track of whether vertex is discovered
    discovered = [False] * N

    # list to store the topological order
    path = []
    allpaths = []
    # find all topological ordering and print them
    findAllTopologicalOrders(graph, path, discovered, N, allpaths, maxnumber=maxnumber)
    return allpaths

# <--- end code section for topological sorts

# find all tasks that depend on a given task (id); when a cache
# dict is given we can fill for the whole graph in one pass...
def find_all_dependent_tasks(possiblenexttask, tid, cache={}):
    c=cache.get(tid)
    if c!=None:
        return c

    daughterlist=[tid]
    # possibly recurse
    for n in possiblenexttask[tid]:
        c = cache.get(n)
        if c == None:
            c = find_all_dependent_tasks(possiblenexttask, n, cache)
        daughterlist = daughterlist + c
        cache[n]=c

    cache[tid]=daughterlist
    return list(set(daughterlist))


# wrapper taking some edges, constructing the graph,
# obtain all topological orderings and some other helper data structures
def analyseGraph(edges, nodes):
    # Number of nodes in the graph
    N = len(nodes)

    # candidate list trivial
    nextjobtrivial = { n:[] for n in nodes }
    # startnodes
    nextjobtrivial[-1] = nodes
    for e in edges:
        nextjobtrivial[e[0]].append(e[1])
        if nextjobtrivial[-1].count(e[1]):
            nextjobtrivial[-1].remove(e[1])

    # find topological orderings of the graph
    # create a graph from edges
    graph = Graph(edges, N)
    orderings = printAllTopologicalOrders(graph)

    return (orderings, nextjobtrivial)


def draw_workflow(workflowspec):
    if not havegraphviz:
        print('graphviz not installed, cannot draw workflow')
        return

    dot = Digraph(comment='Workflow')
    nametoindex={}
    index=0
    # nodes
    for node in workflowspec['stages']:
        name=node['name']
        nametoindex[name]=index
        dot.node(str(index), name)
        index=index+1

    # edges
    for node in workflowspec['stages']:
        toindex = nametoindex[node['name']]
        for req in node['needs']:
            fromindex = nametoindex[req]
            dot.edge(str(fromindex), str(toindex))

    dot.render('workflow.gv')

# builds the graph given a "taskuniverse" list
# builds accompagnying structures tasktoid and idtotask
def build_graph(taskuniverse, workflowspec):
    tasktoid={ t[0]['name']:i for i, t in enumerate(taskuniverse, 0) }
    # print (tasktoid)

    nodes = []
    edges = []
    for t in taskuniverse:
        nodes.append(tasktoid[t[0]['name']])
        for n in t[0]['needs']:
            edges.append((tasktoid[n], tasktoid[t[0]['name']]))

    return (edges, nodes)


# loads the workflow specification
def load_workflow(workflowfile):
    fp=open(workflowfile)
    workflowspec=json.load(fp)
    return workflowspec


# filters the original workflowspec according to wanted targets or labels
# returns a new workflowspec
def filter_workflow(workflowspec, targets=[], targetlabels=[]):
    if len(targets)==0:
       return workflowspec
    if len(targetlabels)==0 and len(targets)==1 and targets[0]=="*":
       return workflowspec

    transformedworkflowspec = workflowspec

    def task_matches(t):
        for filt in targets:
            if filt=="*":
                return True
            if re.match(filt, t)!=None:
                return True
        return False

    def task_matches_labels(t):
        # when no labels are given at all it's ok
        if len(targetlabels)==0:
            return True

        for l in t['labels']:
            if targetlabels.count(l)!=0:
                return True
        return False

    # The following sequence of operations works and is somewhat structured.
    # However, it builds lookups used elsewhere as well, so some CPU might be saved by reusing
    # some structures across functions or by doing less passes on the data.

    # helper lookup
    tasknametoid = { t['name']:i for i, t in enumerate(workflowspec['stages'],0) }

    # build full target list
    full_target_list = [ t for t in workflowspec['stages'] if task_matches(t['name']) and task_matches_labels(t) ]
    full_target_name_list = [ t['name'] for t in full_target_list ]

    # build full dependency list for a task t
    def getallrequirements(t):
        _l=[]
        for r in t['needs']:
            fulltask = workflowspec['stages'][tasknametoid[r]]
            _l.append(fulltask)
            _l=_l+getallrequirements(fulltask)
        return _l

    full_requirements_list = [ getallrequirements(t) for t in full_target_list ]

    # make flat and fetch names only
    full_requirements_name_list = list(set([ item['name'] for sublist in full_requirements_list for item in sublist ]))

    # inner "lambda" helper answering if a task "name" is needed by given targets
    def needed_by_targets(name):
        if full_target_name_list.count(name)!=0:
            return True
        if full_requirements_name_list.count(name)!=0:
            return True
        return False

    # we finaly copy everything matching the targets as well
    # as all their requirements
    transformedworkflowspec['stages']=[ l for l in workflowspec['stages'] if needed_by_targets(l['name']) ]
    return transformedworkflowspec


# builds topological orderings (for each timeframe)
def build_dag_properties(workflowspec):
    globaltaskuniverse = [ (l, i) for i, l in enumerate(workflowspec['stages'], 1) ]
    timeframeset = set( l['timeframe'] for l in workflowspec['stages'] )

    edges, nodes = build_graph(globaltaskuniverse, workflowspec)
    tup = analyseGraph(edges, nodes.copy())
    #
    global_next_tasks = tup[1]


    # a simple score for importance of nodes
    # for each task find number of nodes that depend on a task -> might be weighted with CPU and MEM needs
    importance_score = [ 0 for n in nodes ]
    dependency_cache = {}
    for n in nodes:
        importance_score[n] = len(find_all_dependent_tasks(global_next_tasks, n, dependency_cache))
        actionlogger.info("Score for " + str(globaltaskuniverse[n][0]['name']) + " is " + str(importance_score[n]))

    # weight influences scheduling order can be anything user defined ... for the moment we just prefer to stay within a timeframe
    def getweight(tid):
        return globaltaskuniverse[tid][0]['timeframe']

    task_weights = [ getweight(tid) for tid in range(len(globaltaskuniverse)) ]

    # print (global_next_tasks)
    return { 'nexttasks' : global_next_tasks, 'weights' : task_weights, 'topological_ordering' : tup[0] }


#
# functions for execution; encapsulated in a WorkflowExecutor class
#
class WorkflowExecutor:
    # Constructor
    def __init__(self, workflowfile, args, jmax=100):
      self.args=args
      self.workflowfile = workflowfile
      self.workflowspec = load_workflow(workflowfile)
      self.workflowspec = filter_workflow(self.workflowspec, args.target_tasks, args.target_labels)

      if len(self.workflowspec['stages']) == 0:
          print ('Workflow is empty. Nothing to do')
          exit (0)

      workflow = build_dag_properties(self.workflowspec)
      if args.visualize_workflow:
          draw_workflow(self.workflowspec)
      self.possiblenexttask = workflow['nexttasks']
      self.taskweights = workflow['weights']
      self.topological_orderings = workflow['topological_ordering']
      self.taskuniverse = [ l['name'] for l in self.workflowspec['stages'] ]
      self.idtotask = [ 0 for l in self.taskuniverse ]
      self.tasktoid = {}
      for i in range(len(self.taskuniverse)):
          self.tasktoid[self.taskuniverse[i]]=i
          self.idtotask[i]=self.taskuniverse[i]

      self.maxmemperid = [ self.workflowspec['stages'][tid]['resources']['mem'] for tid in range(len(self.taskuniverse)) ]
      self.cpuperid = [ self.workflowspec['stages'][tid]['resources']['cpu'] for tid in range(len(self.taskuniverse)) ]
      self.curmembooked = 0
      self.curcpubooked = 0
      self.curmembooked_backfill = 0
      self.curcpubooked_backfill = 0
      self.memlimit = float(args.mem_limit) # some configurable number
      self.cpulimit = float(args.cpu_limit)
      self.procstatus = { tid:'ToDo' for tid in range(len(self.workflowspec['stages'])) }
      self.taskneeds= { t:set(self.getallrequirements(t)) for t in self.taskuniverse }
      self.stoponfailure = True
      self.max_jobs_parallel = int(jmax)
      self.scheduling_iteration = 0
      self.process_list = []  # list of currently scheduled tasks with normal priority
      self.backfill_process_list = [] # list of curently scheduled tasks with low backfill priority (not sure this is needed)
      self.pid_to_psutilsproc = {}  # cache of putilsproc for resource monitoring
      self.pid_to_files = {} # we can auto-detect what files are produced by which task (at least to some extent)
      self.pid_to_connections = {} # we can auto-detect what connections are opened by which task (at least to some extent)
      signal.signal(signal.SIGINT, self.SIGHandler)
      signal.siginterrupt(signal.SIGINT, False)
      self.nicevalues = [ os.nice(0) for tid in range(len(self.taskuniverse)) ]
      self.internalmonitorcounter = 0 # internal use
      self.internalmonitorid = 0 # internal use
      self.tids_marked_toretry = [] # sometimes we might want to retry a failed task (simply because it was "unlucky") and we put them here
      self.retry_counter = [ 0 for tid in range(len(self.taskuniverse)) ] # we keep track of many times retried already
      self.semaphore_values = { self.workflowspec['stages'][tid].get('semaphore'):0 for tid in range(len(self.taskuniverse)) if self.workflowspec['stages'][tid].get('semaphore')!=None } # keeps current count of semaphores (defined in the json workflow). used to achieve user-defined "critical sections".

    def SIGHandler(self, signum, frame):
       # basically forcing shut down of all child processes
       actionlogger.info("Signal " + str(signum) + " caught")
       try:
           procs = psutil.Process().children(recursive=True)
       except (psutil.NoSuchProcess):
           pass
       except (psutil.AccessDenied, PermissionError):
           procs = getChildProcs(os.getpid())

       for p in procs:
           actionlogger.info("Terminating " + str(p))
           try:
             p.terminate()
           except (psutil.NoSuchProcess, psutil.AccessDenied):
             pass

       gone, alive = psutil.wait_procs(procs, timeout=3)
       for p in alive:
           try:
             actionlogger.info("Killing " + str(p))
             p.kill()
           except (psutil.NoSuchProcess, psutil.AccessDenied):
             pass

       exit (1)

    def getallrequirements(self, t):
        l=[]
        for r in self.workflowspec['stages'][self.tasktoid[t]]['needs']:
            l.append(r)
            l=l+self.getallrequirements(r)
        return l

    def get_done_filename(self, tid):
        name = self.workflowspec['stages'][tid]['name']
        workdir = self.workflowspec['stages'][tid]['cwd']
        # name and workdir define the "done" file as used by taskwrapper
        # this assumes that taskwrapper is used to actually check if something is to be rerun
        done_filename = workdir + '/' + name + '.log_done'
        return done_filename

    # removes the done flag from tasks that need to be run again
    def remove_done_flag(self, listoftaskids):
       for tid in listoftaskids:
          done_filename = self.get_done_filename(tid)
          name=self.workflowspec['stages'][tid]['name']
          if args.dry_run:
              print ("Would mark task " + name + " as to be done again")
          else:
              print ("Marking task " + name + " as to be done again")
              if os.path.exists(done_filename) and os.path.isfile(done_filename):
                  os.remove(done_filename)

    # submits a task as subprocess and records Popen instance
    def submit(self, tid, nice=os.nice(0)):
      actionlogger.debug("Submitting task " + str(self.idtotask[tid]) + " with nice value " + str(nice))
      c = self.workflowspec['stages'][tid]['cmd']
      workdir = self.workflowspec['stages'][tid]['cwd']
      if not workdir=='':
          if os.path.exists(workdir) and not os.path.isdir(workdir):
                  actionlogger.error('Cannot create working dir ... some other resource exists already')
                  return None

          if not os.path.isdir(workdir):
                  os.mkdir(workdir)

      self.procstatus[tid]='Running'
      if args.dry_run:
          drycommand="echo \' " + str(self.scheduling_iteration) + " : would do " + str(self.workflowspec['stages'][tid]['name']) + "\'"
          return subprocess.Popen(['/bin/bash','-c',drycommand], cwd=workdir)

      taskenv = os.environ.copy()
      # add task specific environment
      if self.workflowspec['stages'][tid].get('env')!=None:
          taskenv.update(self.workflowspec['stages'][tid]['env'])

      p = psutil.Popen(['/bin/bash','-c',c], cwd=workdir, env=taskenv)
      try:
          p.nice(nice)
          self.nicevalues[tid]=nice
      except (psutil.NoSuchProcess, psutil.AccessDenied):
          actionlogger.error('Couldn\'t set nice value of ' + str(p.pid) + ' to ' + str(nice) + ' -- current value is ' + str(p.nice()))
          self.nicevalues[tid]=os.nice(0)
      return p

    def ok_to_submit(self, tid, backfill=False):
      softcpufactor=1
      softmemfactor=1
      if backfill:
          softcpufactor=1.5
          sotmemfactor=1.5

      # check semaphore
      sem = self.workflowspec['stages'][tid].get('semaphore')
      if sem != None:
        if self.semaphore_values[sem] > 0:
           return False

      # check other resources
      if not backfill:
          # analyse CPU
          okcpu = (self.curcpubooked + float(self.cpuperid[tid]) <= self.cpulimit)
          # analyse MEM
          okmem = (self.curmembooked + float(self.maxmemperid[tid]) <= self.memlimit)
          actionlogger.debug ('Condition check --normal-- for  ' + str(tid) + ':' + str(self.idtotask[tid]) + ' CPU ' + str(okcpu) + ' MEM ' + str(okmem))
          return (okcpu and okmem)
      else:
          # not backfilling jobs which either take much memory or use lot's of CPU anyway
          # conditions are somewhat arbitrary and can be played with
          if float(self.cpuperid[tid]) > 0.9*float(self.args.cpu_limit):
              return False
          if float(self.maxmemperid[tid])/float(self.args.cpu_limit) >= 1900:
              return False

          # analyse CPU
          okcpu = (self.curcpubooked_backfill + float(self.cpuperid[tid]) <= self.cpulimit)
          okcpu = okcpu and (self.curcpubooked + self.curcpubooked_backfill + float(self.cpuperid[tid]) <= softcpufactor*self.cpulimit)
          # analyse MEM
          okmem = (self.curmembooked + self.curmembooked_backfill + float(self.maxmemperid[tid]) <= softmemfactor*self.memlimit)
          actionlogger.debug ('Condition check --backfill-- for  ' + str(tid) + ':' + str(self.idtotask[tid]) + ' CPU ' + str(okcpu) + ' MEM ' + str(okmem))
          return (okcpu and okmem)
      return False


    def ok_to_skip(self, tid):
        done_filename = self.get_done_filename(tid)
        if os.path.exists(done_filename) and os.path.isfile(done_filename):
            return True
        return False

    def book_resources(self, tid, backfill = False):
        # books the resources used by a certain task
        # semaphores
        sem = self.workflowspec['stages'][tid].get('semaphore')
        if sem != None:
          self.semaphore_values[sem]+=1

        # CPU + MEM
        if not backfill:
          self.curmembooked+=float(self.maxmemperid[tid])
          self.curcpubooked+=float(self.cpuperid[tid])
        else:
          self.curmembooked_backfill+=float(self.maxmemperid[tid])
          self.curcpubooked_backfill+=float(self.cpuperid[tid])

    def unbook_resources(self, tid, backfill = False):
        # "frees" the nominal resources used by a certain task from the accounting
        # so that other jobs can be scheduled
        sem = self.workflowspec['stages'][tid].get('semaphore')
        if sem != None:
          self.semaphore_values[sem]-=1

        # CPU + MEM
        if not backfill:
          self.curmembooked-=float(self.maxmemperid[tid])
          self.curcpubooked-=float(self.cpuperid[tid])
        else:
          self.curmembooked_backfill-=float(self.maxmemperid[tid])
          self.curcpubooked_backfill-=float(self.cpuperid[tid])


    def try_job_from_candidates(self, taskcandidates, process_list, finished):
       self.scheduling_iteration = self.scheduling_iteration + 1

       # remove "done / skippable" tasks immediately
       tasks_skipped = False
       for tid in taskcandidates.copy():  # <--- the copy is important !! otherwise this loop is not doing what you think
          if self.ok_to_skip(tid):
              finished.append(tid)
              taskcandidates.remove(tid)
              tasks_skipped = True
              actionlogger.info("Skipping task " + str(self.idtotask[tid]))

       # if tasks_skipped:
       #   return # ---> we return early in order to preserve some ordering (the next candidate tried should be daughters of skipped jobs)

       # the ordinary process list part
       initialcandidates=taskcandidates.copy()
       for tid in initialcandidates:
          actionlogger.debug ("trying to submit " + str(tid) + ':' + str(self.idtotask[tid]))
          if (len(self.process_list) + len(self.backfill_process_list) < self.max_jobs_parallel) and self.ok_to_submit(tid):
            p=self.submit(tid)
            if p!=None:
                self.book_resources(tid)
                self.process_list.append((tid,p))
                taskcandidates.remove(tid)
                # minimal delay
                time.sleep(0.1)
          else:
             break #---> we break at first failure assuming some priority (other jobs may come in via backfill)

       # the backfill part for remaining candidates
       initialcandidates=taskcandidates.copy()
       for tid in initialcandidates:
          actionlogger.debug ("trying to backfill submit " + str(tid) + ':' + str(self.idtotask[tid]))

          if (len(self.process_list) + len(self.backfill_process_list) < self.max_jobs_parallel) and self.ok_to_submit(tid, backfill=True):
            p=self.submit(tid, 19)
            if p!=None:
                self.book_resources(tid, backfill=True)
                self.process_list.append((tid,p))
                taskcandidates.remove(tid) #-> not sure about this one
                # minimal delay
                time.sleep(0.1)
          else:
             continue

    def stop_pipeline_and_exit(self, process_list):
        # kill all remaining jobs
        for p in process_list:
           p[1].kill()

        exit(1)

    def monitor(self, process_list):
        self.internalmonitorcounter+=1
        if self.internalmonitorcounter % 5 != 0:
            return

        self.internalmonitorid+=1

        globalCPU=0.
        globalPSS=0.
        globalCPU_backfill=0.
        globalPSS_backfill=0.
        resources_per_task = {}
        for tid, proc in process_list:
            # proc is Popen object
            pid=proc.pid
            if self.pid_to_files.get(pid)==None:
                self.pid_to_files[pid]=set()
                self.pid_to_connections[pid]=set()
            try:
                psutilProcs = [ proc ]
                # use psutil for CPU measurement
                psutilProcs = psutilProcs + proc.children(recursive=True)
            except (psutil.NoSuchProcess):
                continue

            except (psutil.AccessDenied, PermissionError):
                psutilProcs = psutilProcs + getChildProcs(pid)

            # accumulate total metrics (CPU, memory)
            totalCPU = 0.
            totalPSS = 0.
            totalSWAP = 0.
            totalUSS = 0.
            for p in psutilProcs:
                """
                try:
                    for f in p.open_files():
                        self.pid_to_files[pid].add(str(f.path)+'_'+str(f.mode))
                    for f in p.connections(kind="all"):
                        remote=f.raddr
                        if remote==None:
                            remote='none'
                        self.pid_to_connections[pid].add(str(f.type)+"_"+str(f.laddr)+"_"+str(remote))
                except Exception:
                    pass
                """
                thispss=0
                thisuss=0
                # MEMORY part
                try:
                    fullmem=p.memory_full_info()
                    thispss=getattr(fullmem,'pss',0) #<-- pss not available on MacOS
                    totalPSS=totalPSS + thispss
                    totalSWAP=totalSWAP + fullmem.swap
                    thisuss=fullmem.uss
                    totalUSS=totalUSS + thisuss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                # CPU part
                # fetch existing proc or insert
                cachedproc = self.pid_to_psutilsproc.get(p.pid)
                if cachedproc!=None:
                    try:
                        thiscpu = cachedproc.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        thiscpu = 0.
                    totalCPU = totalCPU + thiscpu
                    # thisresource = {'iter':self.internalmonitorid, 'pid': p.pid, 'cpu':thiscpu, 'uss':thisuss/1024./1024., 'pss':thispss/1024./1024.}
                    # metriclogger.info(thisresource)
                else:
                    self.pid_to_psutilsproc[p.pid] = p
                    try:
                        self.pid_to_psutilsproc[p.pid].cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            resources_per_task[tid]={'iter':self.internalmonitorid, 'name':self.idtotask[tid], 'cpu':totalCPU, 'uss':totalUSS/1024./1024., 'pss':totalPSS/1024./1024, 'nice':proc.nice(), 'swap':totalSWAP, 'label':self.workflowspec['stages'][tid]['labels']}
            metriclogger.info(resources_per_task[tid])
            send_webhook(self.args.webhook, resources_per_task)

        for r in resources_per_task.values():
            if r['nice']==os.nice(0):
                globalCPU+=r['cpu']
                globalPSS+=r['pss']
            else:
                globalCPU_backfill+=r['cpu']
                globalPSS_backfill+=r['pss']

        if globalPSS > self.memlimit:
            metriclogger.info('*** MEMORY LIMIT PASSED !! ***')
            # --> We could use this for corrective actions such as killing jobs currently back-filling
            # (or better hibernating)

    def waitforany(self, process_list, finished):
       failuredetected = False
       failingpids = []
       failingtasks = []
       if len(process_list)==0:
           return False

       for p in list(process_list):
          pid = p[1].pid
          tid = p[0]  # the task id of this process
          returncode = 0
          if not self.args.dry_run:
              returncode = p[1].poll()
          if returncode!=None:
            actionlogger.info ('Task ' + str(pid) + ' ' + str(tid)+':'+str(self.idtotask[tid]) + ' finished with status ' + str(returncode))
            # account for cleared resources
            self.unbook_resources(tid, backfill = self.nicevalues[tid]!=os.nice(0) )
            self.procstatus[tid]='Done'
            finished.append(tid)
            process_list.remove(p)
            if returncode != 0:
               print (str(tid) + ' failed ... checking retry')
               # we inspect if this is something "unlucky" which could be resolved by a simple rebsumit
               if self.is_worth_retrying(tid) and self.retry_counter[tid] < 2:
                 print (str(tid) + ' to be retried')
                 actionlogger.info ('Task ' + str(self.idtotask[tid]) + ' failed but marked to be retried ')
                 self.tids_marked_toretry.append(tid)
                 self.retry_counter[tid] += 1

               else:
                 failuredetected = True
                 failingpids.append(pid)
                 failingtasks.append(tid)

       if failuredetected and self.stoponfailure:
          actionlogger.info('Stoping pipeline due to failure in stages with PID ' + str(failingpids))
          # self.analyse_files_and_connections()
          self.cat_logfiles_tostdout(failingtasks)
          self.send_checkpoint(failingtasks, self.args.checkpoint_on_failure)
          self.stop_pipeline_and_exit(process_list)

       # empty finished means we have to wait more
       return len(finished)==0


    def get_logfile(self, tid):
        # determines the logfile name for this task
        taskspec = self.workflowspec['stages'][tid]
        taskname = taskspec['name']
        filename = taskname + '.log'
        directory = taskspec['cwd']
        return directory + '/' + filename


    def is_worth_retrying(self, tid):
        # This checks for some signatures in logfiles that indicate that a retry of this task
        # might have a chance.
        # Ideally, this should be made user configurable. Either the user could inject a lambda
        # or a regular expression to use. For now we just put a hard coded list
        logfile = self.get_logfile(tid)

        # 1) ZMQ_EVENT + interrupted system calls (DPL bug during shutdown)
        # Not sure if grep is faster than native Python text search ...
        status = os.system('grep "failed setting ZMQ_EVENTS" ' + logfile + ' &> /dev/null')
        if os.WEXITSTATUS(status) == 0:
           return True

        return False


    def cat_logfiles_tostdout(self, taskids):
        # In case of errors we can cat the logfiles for this taskname
        # to stdout. Assuming convention that "taskname" translates to "taskname.log" logfile.
        for tid in taskids:
            logfile = self.get_logfile(tid)
            if os.path.exists(logfile):
                print (' ----> START OF LOGFILE ', logfile, ' -----')
                os.system('cat ' + logfile)
                print (' <---- END OF LOGFILE ', logfile, ' -----')

    def send_checkpoint(self, taskids, location):
        # Makes a tarball containing all files in the base dir
        # (timeframe independent) and the dir with corrupted timeframes
        # and copies it to a specific ALIEN location. Not are core function
        # just some tool get hold on error conditions appearing on the GRID.

        def get_tar_command(dir='./', flags='cf', filename='checkpoint.tar'):
            return 'find ' + str(dir) + ' -maxdepth 1 -type f -print0 | xargs -0 tar ' + str(flags) + ' ' + str(filename)

        if location != None:
           print ('Making a failure checkpoint')
           # let's determine a filename from ALIEN_PROC_ID - hostname - and PID

           aliprocid=os.environ.get('ALIEN_PROC_ID')
           if aliprocid == None:
              aliprocid = 0

           fn='pipeline_checkpoint_ALIENPROC' + str(aliprocid) + '_PID' + str(os.getpid()) + '_HOST' + socket.gethostname() + '.tar'
           actionlogger.info("Checkpointing to file " + fn)
           tarcommand = get_tar_command(filename=fn)
           actionlogger.info("Taring " + tarcommand)

           # first of all the base directory
           os.system(tarcommand)
           # then we add stuff for the specific timeframes ids if any
           for tid in taskids:
             taskspec = self.workflowspec['stages'][tid]
             directory = taskspec['cwd']
             if directory != "./":
               tarcommand = get_tar_command(dir=directory, flags='rf', filename=fn)
               actionlogger.info("Tar command is " + tarcommand)
               os.system(tarcommand)

           # location needs to be an alien path of the form alien:///foo/bar/
           copycommand='alien.py cp ' + fn + ' ' + str(location) + '@disk:1'
           actionlogger.info("Copying to alien " + copycommand)
           os.system(copycommand)


    def analyse_files_and_connections(self):
        for p,s in self.pid_to_files.items():
            for f in s:
                print("F" + str(f) + " : " + str(p))
        for p,s in self.pid_to_connections.items():
            for c in s:
               print("C" + str(c) + " : " + str(p))
            #print(str(p) + " CONS " + str(c))
        try:
            # check for intersections
            for p1, s1 in self.pid_to_files.items():
                for p2, s2 in self.pid_to_files.items():
                    if p1!=p2:
                        if type(s1) is set and type(s2) is set:
                            if len(s1)>0 and len(s2)>0:
                                try:
                                    inters = s1.intersection(s2)
                                except Exception:
                                    print ('Exception during intersect inner')
                                    pass
                                if (len(inters)>0):
                                    print ('FILE Intersection ' + str(p1) + ' ' + str(p2) + ' ' + str(inters))
          # check for intersections
            for p1, s1 in self.pid_to_connections.items():
                for p2, s2 in self.pid_to_connections.items():
                    if p1!=p2:
                        if type(s1) is set and type(s2) is set:
                            if len(s1)>0 and len(s2)>0:
                                try:
                                    inters = s1.intersection(s2)
                                except Exception:
                                    print ('Exception during intersect inner')
                                    pass
                                if (len(inters)>0):
                                    print ('CON Intersection ' + str(p1) + ' ' + str(p2) + ' ' + str(inters))

            # check for intersections
            #for p1, s1 in slf.pid_to_files.items():
            #    for p2, s2 in self.pid_to_files.items():
            #        if p1!=p2 and len(s1.intersection(s2))!=0:
            #            print ('Intersection found files ' + str(p1) + ' ' + str(p2) + ' ' + s1.intersection(s2))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('Exception during intersect outer')
            pass

    def is_good_candidate(self, candid, finishedtasks):
        if self.procstatus[candid] != 'ToDo':
            return False
        needs = set([self.tasktoid[t] for t in self.taskneeds[self.idtotask[candid]]])
        if set(finishedtasks).intersection(needs) == needs:
            return True
        return False

    def emit_code_for_task(self, tid, lines):
        actionlogger.debug("Submitting task " + str(self.idtotask[tid]))
        taskspec = self.workflowspec['stages'][tid]
        c = taskspec['cmd']
        workdir = taskspec['cwd']
        env = taskspec.get('env')
        # in general:
        # try to make folder
        lines.append('[ ! -d ' + workdir + ' ] && mkdir ' + workdir + '\n')
        # cd folder
        lines.append('cd ' + workdir + '\n')
        # set local environment
        if env!=None:
            for e in env.items():
                lines.append('export ' + e[0] + '=' + str(e[1]) + '\n')
        # do command
        lines.append(c + '\n')
        # unset local environment
        if env!=None:
            for e in env.items():
                lines.append('unset ' + e[0] + '\n')

        # cd back
        lines.append('cd $OLDPWD\n')


    # produce a bash script that runs workflow standalone
    def produce_script(self, filename):
        # pick one of the correct task orderings
        taskorder = self.topological_orderings[0]
        outF = open(filename, "w")

        lines=[]
        # header
        lines.append('#!/usr/bin/env bash\n')
        lines.append('#THIS FILE IS AUTOGENERATED\n')
        lines.append('JOBUTILS_SKIPDONE=ON\n')
        for tid in taskorder:
            print ('Doing task ' + self.idtotask[tid])
            self.emit_code_for_task(tid, lines)

        outF.writelines(lines)
        outF.close()


    def execute(self):
        psutil.cpu_percent(interval=None)
        os.environ['JOBUTILS_SKIPDONE'] = "ON"

        # we make our own "tmp" folder
        # where we can put stuff such as tmp socket files etc (for instance DPL FAIR-MQ sockets)
        # (In case of running within docker/singularity, this may not be so important)
        if not os.path.isdir("./.tmp"):
          os.mkdir("./.tmp")
        if os.environ.get('FAIRMQ_IPC_PREFIX')==None:
          socketpath = os.getcwd() + "/.tmp"
          actionlogger.info("Setting FAIRMQ socket path to " + socketpath)
          os.environ['FAIRMQ_IPC_PREFIX'] = socketpath

        # some maintenance / init work
        if args.list_tasks:
          print ('List of tasks in this workflow:')
          for i,t in enumerate(self.workflowspec['stages'],0):
              print (t['name'] + '  (' + str(t['labels']) + ')' + ' ToDo: ' + str(not self.ok_to_skip(i)))
          exit (0)

        if args.produce_script != None:
          self.produce_script(args.produce_script)
          exit (0)

        if args.rerun_from:
          reruntaskfound=False
          for task in self.workflowspec['stages']:
              taskname=task['name']
              if re.match(args.rerun_from, taskname):
                reruntaskfound=True
                taskid=self.tasktoid[taskname]
                self.remove_done_flag(find_all_dependent_tasks(self.possiblenexttask, taskid))
          if not reruntaskfound:
              print('No task matching ' + args.rerun_from + ' found; cowardly refusing to do anything ')
              exit (1)

        # *****************
        # main control loop
        # *****************
        currenttimeframe=1
        candidates = [ tid for tid in self.possiblenexttask[-1] ]

        self.process_list=[] # list of tuples of nodes ids and Popen subprocess instances

        finishedtasks=[] # global list of finished tasks
        try:

            while True:
                # sort candidate list according to task weights
                candidates = [ (tid, self.taskweights[tid]) for tid in candidates ]
                candidates.sort(key=lambda tup: tup[1])
                # remove weights
                candidates = [ tid for tid,_ in candidates ]

                finished = [] # --> to account for finished because already done or skipped
                actionlogger.debug('Sorted current candidates: ' + str([(c,self.idtotask[c]) for c in candidates]))
                self.try_job_from_candidates(candidates, self.process_list, finished)
                if len(candidates) > 0 and len(self.process_list) == 0:
                    actionlogger.info("Not able to make progress: Nothing scheduled although non-zero candidate set")
                    send_webhook(self.args.webhook,"Unable to make further progress: Quitting")
                    break

                finished_from_started = [] # to account for finished when actually started
                while self.waitforany(self.process_list, finished_from_started):
                    if not args.dry_run:
                        self.monitor(self.process_list) #  ---> make async to normal operation?
                        time.sleep(1) # <--- make this incremental (small wait at beginning)
                    else:
                        time.sleep(0.001)

                finished = finished + finished_from_started
                actionlogger.debug("finished now :" + str(finished_from_started))
                finishedtasks = finishedtasks + finished

                # if a task was marked as "retry" we simply put it back into the candidate list
                if len(self.tids_marked_toretry) > 0:
                    candidates = candidates + self.tids_marked_toretry
                    self.tids_marked_toretry = []

                # new candidates
                for tid in finished:
                    if self.possiblenexttask.get(tid)!=None:
                        potential_candidates=list(self.possiblenexttask[tid])
                        for candid in potential_candidates:
                        # try to see if this is really a candidate:
                            if self.is_good_candidate(candid, finishedtasks) and candidates.count(candid)==0:
                                candidates.append(candid)

                actionlogger.debug("New candidates " + str( candidates))
                send_webhook(self.args.webhook, "New candidates " + str(candidates))

                if len(candidates)==0 and len(self.process_list)==0:
                   break
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            traceback.print_exc()
            print ('Cleaning up ')

            self.SIGHandler(0,0)

        print ('\n**** Pipeline done *****\n')
        # self.analyse_files_and_connections()

import argparse
import psutil
max_system_mem=psutil.virtual_memory().total

parser = argparse.ArgumentParser(description='Parallel execution of a (O2-DPG) DAG data/job pipeline under resource contraints.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f','--workflowfile', help='Input workflow file name', required=True)
parser.add_argument('-jmax','--maxjobs', help='Number of maximal parallel tasks.', default=100)
parser.add_argument('--dry-run', action='store_true', help='Show what you would do.')
parser.add_argument('--visualize-workflow', action='store_true', help='Saves a graph visualization of workflow.')
parser.add_argument('--target-labels', nargs='+', help='Runs the pipeline by target labels (example "TPC" or "DIGI").\
                    This condition is used as logical AND together with --target-tasks.', default=[])
parser.add_argument('-tt','--target-tasks', nargs='+', help='Runs the pipeline by target tasks (example "tpcdigi"). By default everything in the graph is run. Regular expressions supported.', default=["*"])
parser.add_argument('--produce-script', help='Produces a shell script that runs the workflow in serialized manner and quits.')
parser.add_argument('--rerun-from', help='Reruns the workflow starting from given task (or pattern). All dependent jobs will be rerun.')
parser.add_argument('--list-tasks', help='Simply list all tasks by name and quit.', action='store_true')

parser.add_argument('--mem-limit', help='Set memory limit as scheduling constraint', default=max_system_mem)
parser.add_argument('--cpu-limit', help='Set CPU limit (core count)', default=8)
parser.add_argument('--cgroup', help='Execute pipeline under a given cgroup (e.g., 8coregrid) emulating resource constraints. This m\
ust exist and the tasks file must be writable to with the current user.')
parser.add_argument('--stdout-on-failure', action='store_true', help='Print log files of failing tasks to stdout,')
parser.add_argument('--webhook', help=argparse.SUPPRESS) # log some infos to this webhook channel
parser.add_argument('--checkpoint-on-failure', help=argparse.SUPPRESS) # debug option making a debug-tarball and sending to specified address
                                                                       # argument is alien-path

args = parser.parse_args()
print (args)

if args.cgroup!=None:
    myPID=os.getpid()
    command="echo " + str(myPID) + " > /sys/fs/cgroup/cpuset/"+args.cgroup+"/tasks"
    actionlogger.info("applying cgroups " + command)
    os.system(command)

executor=WorkflowExecutor(args.workflowfile,jmax=args.maxjobs,args=args)
executor.execute()
