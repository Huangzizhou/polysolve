#ifdef POLYSOLVE_WITH_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverHypreGMRES.hpp>

#include <HYPRE_krylov.h>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverHypreGMRES::LinearSolverHypreGMRES()
    {
#ifdef MPI_VERSION
        /* Initialize MPI */
        int argc = 1;
        char name[] = "";
        char *argv[] = {name};
        char **argvv = &argv[0];
        int myid, num_procs;
        MPI_Init(&argc, &argvv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif
    }

    // Set solver parameters
    void LinearSolverHypreGMRES::setParameters(const json &params)
    {
        if (params.count("max_iter"))
        {
            max_iter_ = params["max_iter"];
        }
        if (params.count("pre_max_iter"))
        {
            pre_max_iter_ = params["pre_max_iter"];
        }

        if (params.count("conv_tol"))
        {
            conv_tol_ = params["conv_tol"];
        }
        else if (params.count("tolerance"))
        {
            conv_tol_ = params["tolerance"];
        }
    }

    void LinearSolverHypreGMRES::getInfo(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverHypreGMRES::analyzePattern(const StiffnessMatrix &Ain, const int precond_num)
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        has_matrix_ = true;
        const HYPRE_Int rows = Ain.rows();
        const HYPRE_Int cols = Ain.cols();

        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, rows - 1, 0, cols - 1, &A);
        // HYPRE_IJMatrixSetPrintLevel(A, 2);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        // HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);

        // TODO: More efficient initialization of the Hypre matrix?
        for (HYPRE_Int k = 0; k < Ain.outerSize(); ++k)
        {
            for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it)
            {
                const HYPRE_Int i[1] = {it.row()};
                const HYPRE_Int j[1] = {it.col()};
                const HYPRE_Complex v[1] = {it.value()};
                HYPRE_Int n_cols[1] = {1};

                HYPRE_IJMatrixSetValues(A, 1, n_cols, i, j, v);
            }
        }

        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    }

    ////////////////////////////////////////////////////////////////////////////////
    
    namespace
    {

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 8;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }
    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverHypreGMRES::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        HYPRE_IJVector b;
        HYPRE_ParVector par_b;
        HYPRE_IJVector x;
        HYPRE_ParVector par_x;

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size() - 1, &b);
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b);

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size() - 1, &x);
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x);

        assert(result.size() == rhs.size());

        for (HYPRE_Int i = 0; i < rhs.size(); ++i)
        {
            const HYPRE_Int index[1] = {i};
            const HYPRE_Complex v[1] = {HYPRE_Complex(rhs(i))};
            const HYPRE_Complex z[1] = {HYPRE_Complex(result(i))};

            HYPRE_IJVectorSetValues(b, 1, index, v);
            HYPRE_IJVectorSetValues(x, 1, index, z);
        }

        HYPRE_IJVectorAssemble(b);
        HYPRE_IJVectorGetObject(b, (void **)&par_b);

        HYPRE_IJVectorAssemble(x);
        HYPRE_IJVectorGetObject(x, (void **)&par_x);

        /* GMRES */

        /* Create solver */
        HYPRE_Solver solver, precond;
        HYPRE_ParCSRLGMRESCreate(MPI_COMM_WORLD, &solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_ParCSRLGMRESSetMaxIter(solver, max_iter_); /* max iterations */
        HYPRE_ParCSRLGMRESSetTol(solver, conv_tol_);     /* conv. tolerance */
        // HYPRE_ParCSRLGMRESSetStopCrit(solver, 1);
        // HYPRE_ParCSRLGMRESSetPrintLevel(solver, 2); /* print solve info */
        // HYPRE_ParCSRLGMRESSetLogging(solver, 1); /* needed to get run info later */

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);

#if 0
        //HYPRE_BoomerAMGSetPrintLevel(precond, 2); /* print amg solution info */
        HYPRE_BoomerAMGSetCoarsenType(precond, 6);
        HYPRE_BoomerAMGSetOldDefault(precond);
        HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
        HYPRE_BoomerAMGSetNumSweeps(precond, 1);
        HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
        HYPRE_BoomerAMGSetMaxIter(precond, pre_max_iter_); /* do only one iteration! */
#endif

        HypreBoomerAMG_SetDefaultOptions(precond);

        /* Set the preconditioner */
        HYPRE_ParCSRLGMRESSetPrecond(solver, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, precond);

        /* Now setup and solve! */
        HYPRE_ParCSRLGMRESSetup(solver, parcsr_A, par_b, par_x);
        HYPRE_ParCSRLGMRESSolve(solver, parcsr_A, par_b, par_x);

        /* Run info - needed logging turned on */
        // HYPRE_ParCSRLGMRESGetNumIterations(solver, &num_iterations);
        // HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

        // printf("\n");
        // printf("Iterations = %lld\n", num_iterations);
        // printf("Final Relative Residual Norm = %g\n", final_res_norm);
        // printf("\n");

        /* Destroy solver and preconditioner */
        HYPRE_ParCSRLGMRESDestroy(solver);

        assert(result.size() == rhs.size());
        for (HYPRE_Int i = 0; i < rhs.size(); ++i)
        {
            const HYPRE_Int index[1] = {i};
            HYPRE_Complex v[1];
            HYPRE_IJVectorGetValues(x, 1, index, v);

            result(i) = v[0];
        }

        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverHypreGMRES::~LinearSolverHypreGMRES()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
    }

} // namespace polysolve

#endif
