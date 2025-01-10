from multiprocessing import Process


class LazySplittingProcess(Process):
    """
    This splitting process adds jobs to the queue as soon as they become available,
    instead of computing first all the jobs
    """
    TOTAL_JOBS_NUMBER_STRING = "Total_jobs_n"

    def __init__(self, aes_encoder, formula, jobs_queue, reporting_queue, print=False):
        super(LazySplittingProcess, self).__init__()

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

    def add_job(self, job_count, constraints_stack, trace_variables):
        # make list from the stack, a list of lists
        constraints = [item for sublist in constraints_stack for item in sublist]

        # if job_count == 1:
        # print("Number of constraints", len(constraints))
        # print("\n".join([str(c) for c in constraints]))
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        self.jobs_queue.put((job_count, self.aes_encoder.constrs_manager, constraints, trace_variables))

    def compute_splits(self):

        self.aes_encoder.set_splitting_process(self)

        jobs_count = self.formula.acceptI(self.aes_encoder)

        self.reporting_queue.put((jobs_count - 1, self.TOTAL_JOBS_NUMBER_STRING, None, None))
        if self.PRINT_TO_CONSOLE:
            print("Splitting process created", jobs_count - 1, "jobs")
