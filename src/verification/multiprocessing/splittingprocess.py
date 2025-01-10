from multiprocessing import Process


class SplittingProcess(Process):
    TOTAL_JOBS_NUMBER_STRING = "Total_jobs_n"

    def __init__(self, aes_encoder, formula, jobs_queue, reporting_queue, print=False):
        super(SplittingProcess, self).__init__()

        """
        """
        self.aes_encoder = aes_encoder

        self.formula = formula

        # the queue to which jobs will be added
        """
        Job descriptions are vmodels themselves without gmodel
        so that it could be serialised and shared through a queue
        """
        self.jobs_queue = jobs_queue

        # the queue to communicate with the main process
        # splitter with report to it the total number of splits once they all have been computed
        self.reporting_queue = reporting_queue

        self.PRINT_TO_CONSOLE = print
        self.DEBUG_MODE = False

    def run(self):

        if self.PRINT_TO_CONSOLE:
            print("running splitting process")

        self.compute_splits()

        if self.PRINT_TO_CONSOLE:
            print("Splitting process", "finished")

    def compute_splits(self):
        jobs = self.formula.acceptI(self.aes_encoder)

        job_count = 1
        for job in jobs:
            self.jobs_queue.put((job_count, self.aes_encoder.constrs_manager, job,
                                 self.aes_encoder.constrs_manager.get_variable_tracker().get_trace()))

            # if job_count == 1:
            # print("Number of constraints", len(job))
            # print("\n".join([str(c) for c in job]))
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

            job_count += 1

        self.reporting_queue.put((len(jobs), self.TOTAL_JOBS_NUMBER_STRING, None, None))
