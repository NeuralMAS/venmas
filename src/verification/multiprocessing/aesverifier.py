import multiprocessing as mp
from multiprocessing import Process
import queue
from timeit import default_timer as timer

from src.verification.multiprocessing.lazy_splitting_process import LazySplittingProcess
from src.verification.multiprocessing.splittingprocess import SplittingProcess


class WorkerVerificationProcess(Process):
    TIMEOUT = 14400

    def __init__(self, id, jobs_queue, reporting_queue, print=False):
        super(WorkerVerificationProcess, self).__init__()

        self.id = id
        self.jobs_queue = jobs_queue
        self.reporting_queue = reporting_queue

        self.PRINT_TO_CONSOLE = print

    def run(self):
        while True:
            try:
                job_id, constraint_manager, lin_program, trace_variables = self.jobs_queue.get(timeout=self.TIMEOUT)

                start = timer()
                gmodel = constraint_manager.create_gmodel(lin_program, trace_variables)
                res = constraint_manager.check_feasibility(gmodel, trace_variables=trace_variables)
                end = timer()
                runtime = end - start

                # print("\n".join([str(c) for c in lin_program]))
                # print("-----------------------------------")
                extra = []
                if res == "True":
                    stats = constraint_manager.get_stats()
                    depth = len(stats.witness_states)
                    for i in range(0, depth - 1):
                        extra.append(stats.witness_states[i])
                        extra.append(stats.witness_actions[i])
                        # extra += "state{}: {}\n".format(i, stats.witness_states[i])
                        # extra += "action{}: {}\n".format(i, stats.witness_actions[i])
                    extra.append(stats.witness_states[depth - 1])
                    # extra += "state{}: {}\n".format(depth - 1, stats.witness_states[depth - 1])

                if self.PRINT_TO_CONSOLE:
                   print("Subprocess", self.id, "finished job", job_id, "result:", res, "in", runtime)
                self.reporting_queue.put((job_id, res, runtime, extra))

            except queue.Empty:
                # to handle the case when the main process got killed,
                # but the workers remained alive.
                break


class AESVerifier:

    TIME_LIMIT = 3600

    def __init__(self, aes_encoder, formula, parallel_processes_number, print_to_console=True):
        super(AESVerifier, self).__init__()

        self.parallel_processes_number = parallel_processes_number
        # the queue to which all worker processes report the results
        # and the splitting process will store the total number of splits
        self.reporting_queue = mp.Queue()

        jobs_queue = mp.Queue()

        # from src.verification.complete.verifier.breadth_first_compositional_ctl_milp_encoder import \
        #     CompositionalCTLMILPEncoder
        from src.verification.specencoder.depth_first_compositional_single_agent_ex_milp_encoder import DepthFirstCompositionalSingleAgentExistsNextMILPEncoder
        from src.verification.specencoder.depth_first_compositional_multi_agent_ex_milp_encoder import \
            DepthFirstCompositionalMultiAgentExistsNextMILPEncoder
        if isinstance(aes_encoder, DepthFirstCompositionalSingleAgentExistsNextMILPEncoder) or isinstance(aes_encoder, DepthFirstCompositionalMultiAgentExistsNextMILPEncoder):
            self.splitting_process = LazySplittingProcess(aes_encoder, formula, jobs_queue, self.reporting_queue, print_to_console)
        else:#if isinstance(aes_encoder, CompositionalCTLMILPEncoder):
            self.splitting_process = SplittingProcess(aes_encoder, formula, jobs_queue, self.reporting_queue, print_to_console)

        self.worker_processes = [WorkerVerificationProcess(i+1, jobs_queue, self.reporting_queue, print_to_console)
                                 for i in range(self.parallel_processes_number)]

        self.PRINT_TO_CONSOLE = print_to_console

    def verify(self):

        start = timer()

        # start the splitting and worker processes
        self.splitting_process.start()
        for proc in self.worker_processes:
            proc.start()

        timedout_jobs_count = 0
        finished_jobs_count = 0

        total_number_of_splits = -1

        """ 
        Read results from the reporting queue
        until encountered a True result, or
        until all the splits have completed
        """
        while True:
            try:
                job_id, res, runtime, extra = self.reporting_queue.get(timeout=self.TIME_LIMIT - (timer() - start))

                if res == "True":
                    if self.PRINT_TO_CONSOLE:
                        print("Main process: read True. Terminating...")
                    result = ("True", "{}-{}".format(job_id, finished_jobs_count+1), extra)
                    break

                elif res == "False":
                    finished_jobs_count += 1

                elif res == "Timeout":
                    finished_jobs_count += 1
                    timedout_jobs_count += 1

                elif res == SplittingProcess.TOTAL_JOBS_NUMBER_STRING:
                    # update the total_number of splits
                    total_number_of_splits = job_id
                else:
                    raise Exception("Unexpected result read from reporting queue", res)

                # stopping conditions
                if total_number_of_splits != -1 and finished_jobs_count >= total_number_of_splits:
                    print("Main process: all subproblems have finished. Terminating...")
                    if timedout_jobs_count == 0:
                        result = ("False", total_number_of_splits, None)
                    else:
                        result = ("Timeout", total_number_of_splits, None)
                    break

            except queue.Empty:
                # Timeout occured
                result = ("Timeout", finished_jobs_count, None)
                break
            except KeyboardInterrupt:
                # Received terminating signal
                result = ("Interrupted", finished_jobs_count, None)
                break

        """
        Terminate the splitting and worker processes.
        Especially relevant if there was one early True result.
        """
        try:
            self.splitting_process.terminate()
            for proc in self.worker_processes:
                proc.terminate()
        except:
            print("Error while attempting to terminate processes")

        return result
