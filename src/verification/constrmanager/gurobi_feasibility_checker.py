from gurobipy import *

from src.verification.constrmanager.aes_variable_tracker import AESVariableTracker


class GurobiFeasibilityChecker:
    def __init__(self):
        pass

    @staticmethod
    def check_feasibility(gmodel, stats=None, aes_var_tracker=None, trace_variables=None):
        # :side-effects: Updates, optimizes and reads from Gurobi model, modifies stats object.

        # gmodel.write("program.lp")

        gmodel.update()
        # print("Vars", gmodel.NumVars, "Constrs", gmodel.NumConstrs+gmodel.NumGenConstrs)

        gmodel.optimize()

        if not stats is None:
            # update statistics
            stats.update(gmodel.NumVars, gmodel.NumConstrs+gmodel.NumGenConstrs)

        status = gmodel.status
        if status == GRB.OPTIMAL:
            # Feasible solution found.

            if not stats is None and not trace_variables is None:
                # save the counter-example to the stats object
                # A hack for parallel depth-first execution
                stats.set_witnesses(AESVariableTracker.get_variable_values(gmodel, trace_variables[0]),
                                    AESVariableTracker.get_variable_values(gmodel, trace_variables[1]))

            elif not stats is None and not aes_var_tracker is None:
                # save the counter-example to the stats object
                stats.set_witnesses(aes_var_tracker.get_witness_states(gmodel), aes_var_tracker.get_witness_actions(gmodel))
            return "True"
        elif status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
            ## IF YOU UNCOMMENT, MAKE SURE YOU COMMENT IT FOR THE EXPERIMENTS!!!
            # print("HAVE YOU FORGOTTEN TO COMMENT OUT IIS COMPUTATION???")
            # gmodel.computeIIS()
            # gmodel.write("infeasprogram.ilp")
            return "False"
        elif status == GRB.TIME_LIMIT:
            return "Timeout"
        elif status == GRB.INTERRUPTED:
            return "Interrupted"
        else:
            raise Exception("Unexpected result of solving a LP ", status)

