
#include "Solver.hpp"

#include "descent_strategies/BFGS.hpp"
#include "descent_strategies/Newton.hpp"
#include "descent_strategies/GradientDescent.hpp"
#include "descent_strategies/LBFGS.hpp"

#include <polysolve/Utils.hpp>

#include <jse/jse.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/color.h>
#include <spdlog/fmt/ostr.h>

#include <iomanip>
#include <fstream>

namespace polysolve::nonlinear
{

    // Static constructor
    std::unique_ptr<Solver> Solver::create(
        const json &solver_params_in,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const bool strict_validation)
    {
        json solver_params = solver_params_in; // mutable copy

        json rules;
        jse::JSE jse;

        jse.strict = strict_validation;
        const std::string input_spec = POLYSOLVE_NON_LINEAR_SPEC;
        std::ifstream file(input_spec);

        if (file.is_open())
            file >> rules;
        else
            log_and_throw_error(logger, "unable to open {} rules", input_spec);

        const bool valid_input = jse.verify_json(solver_params, rules);

        if (!valid_input)
            log_and_throw_error(logger, "invalid input json:\n{}", jse.log2str());

        solver_params = jse.inject_defaults(solver_params, rules);

        const std::string solver_name = solver_params["solver"];

        auto solver = std::make_unique<Solver>(solver_name, solver_params, characteristic_length, logger);

        if (solver_name == "BFGS")
        {
            solver->add_strategy(std::make_unique<BFGS>(
                solver_params, linear_solver_params,
                characteristic_length, logger));
        }
        else if (solver_name == "DenseNewton" || solver_name == "dense_newton")
        {
            auto n = Newton::create_solver(false, solver_params, linear_solver_params, characteristic_length, logger);
            for (auto &s : n)
                solver->add_strategy(s);
        }
        else if (solver_name == "Newton" || solver_name == "SparseNewton" || solver_name == "sparse_newton")
        {
            auto n = Newton::create_solver(true, solver_params, linear_solver_params, characteristic_length, logger);
            for (auto &s : n)
                solver->add_strategy(s);
        }
        else if (solver_name == "LBFGS" || solver_name == "L-BFGS")
        {
            solver->add_strategy(std::make_unique<LBFGS>(
                solver_params, characteristic_length, logger));
        }
        else if (solver_name == "GradientDescent" || solver_name == "gradient_descent")
        {
            // grad descent always there
        }
        else
            throw std::runtime_error("Unrecognized solver type: " + solver_name);

        solver->add_strategy(std::make_unique<GradientDescent>(
            solver_params, characteristic_length, logger));

        solver->set_strategies_iterations(solver_params);
        return solver;
    }

    std::vector<std::string> Solver::available_solvers()
    {
        return {"BFGS",
                "DenseNewton",
                "Newton",
                "GradientDescent",
                "L-BFGS"};
    }

    Solver::Solver(const std::string &name,
                   const json &solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger)
        : m_logger(logger), m_name(name), characteristic_length(characteristic_length)
    {
        TCriteria criteria = TCriteria::defaults();
        criteria.xDelta = solver_params["x_delta"];
        criteria.fDelta = solver_params["advanced"]["f_delta"];
        criteria.gradNorm = solver_params["grad_norm"];

        criteria.xDelta *= characteristic_length;
        criteria.fDelta *= characteristic_length;
        criteria.gradNorm *= characteristic_length;

        criteria.iterations = solver_params["max_iterations"];
        // criteria.condition = solver_params["condition"];
        this->setStopCriteria(criteria);

        use_grad_norm_tol = solver_params["line_search"]["use_grad_norm_tol"];
		solver_info_log = solver_params["solver_info_log"];
		export_energy_path = solver_params["export_energy"];

        first_grad_norm_tol = solver_params["first_grad_norm_tol"];

		finite_diff_eps = solver_params["debug_fd_eps"];

        use_grad_norm_tol *= characteristic_length;
        first_grad_norm_tol *= characteristic_length;

        f_delta_step_tol = solver_params["advanced"]["f_delta_step_tol"];

        m_descent_strategy = 0;

        set_line_search(solver_params);
    }

    void Solver::set_strategies_iterations(const json &solver_params)
    {
        m_iter_per_strategy.assign(m_strategies.size() + 1, 1);
        if (solver_params["iterations_per_strategy"].is_array())
        {
            m_iter_per_strategy.resize(m_strategies.size() + 1);
            if (solver_params["iterations_per_strategy"].size() != m_iter_per_strategy.size())
                log_and_throw_error(m_logger, "Invalit iter_per_strategy size: {}!={}", solver_params["iterations_per_strategy"].size(), m_iter_per_strategy.size());

            m_iter_per_strategy = solver_params["iterations_per_strategy"].get<std::vector<int>>();
        }
        else
            m_iter_per_strategy.assign(m_strategies.size() + 1, solver_params["iterations_per_strategy"].get<int>());
    }

    double Solver::compute_grad_norm(const Eigen::VectorXd &x, const Eigen::VectorXd &grad) const
    {
        return grad.norm();
    }

    void Solver::set_line_search(const json &params)
    {
        m_line_search = line_search::LineSearch::create(params, m_logger);
        solver_info["line_search"] = params["line_search"]["method"];
    }

	bool Solver::verify_gradient(Problem &objFunc, const TVector &x, const TVector &grad)
	{
		if (finite_diff_eps <= 0)
			return true;
		
		Eigen::VectorXd direc = grad.normalized();
		Eigen::VectorXd x2 = x + direc * finite_diff_eps;
		Eigen::VectorXd x1 = x - direc * finite_diff_eps;

		objFunc.solution_changed(x2);
		double J2 = objFunc.value(x2);

		objFunc.solution_changed(x1);
		double J1 = objFunc.value(x1);

		double fd = (J2 - J1) / 2 / finite_diff_eps;
		double analytic = direc.dot(grad);

		bool match = abs(fd - analytic) < 1e-8 || abs(fd - analytic) < 1e-1 * abs(analytic);

		// Log error in either case to make it more visible in the logs.
		if (match)
			m_logger.info("step size: {}, finite difference: {}, derivative: {}", finite_diff_eps, fd, analytic);
		else
			m_logger.error("step size: {}, finite difference: {}, derivative: {}", finite_diff_eps, fd, analytic);

		objFunc.solution_changed(x);

		return match;
	}

    void Solver::minimize(Problem &objFunc, TVector &x)
    {
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

        int previous_strategy = m_descent_strategy;
        int current_strategy_iter = 0;
        // ---------------------------
        // Initialize the minimization
        // ---------------------------
        reset(x.size()); // place for children to initialize their fields

        m_line_search->use_grad_norm_tol = use_grad_norm_tol;

        TVector grad = TVector::Zero(x.rows());
        TVector delta_x = TVector::Zero(x.rows());

        // Set these to nan to indicate they have not been computed yet
        double old_energy = NaN;
        {
            POLYSOLVE_SCOPED_STOPWATCH("constraint set update", constraint_set_update_time, m_logger);
            objFunc.solution_changed(x);
        }

        const auto g_norm_tol = this->m_stop.gradNorm;
        this->m_stop.gradNorm = first_grad_norm_tol;

        StopWatch stop_watch("non-linear solver", this->total_time, m_logger);
        stop_watch.start();

		std::ofstream outfile;
		if (export_energy_path != "")
			outfile.open(export_energy_path);

		objFunc.save_to_file(x);

        m_logger.debug(
            "Starting {} with {} solve f₀={:g} ‖∇f₀‖={:g} "
            "(stopping criteria: max_iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})",
            name(), m_line_search->name(),
            objFunc.value(x), this->m_current.gradNorm, this->m_stop.iterations,
            this->m_stop.fDelta, this->m_stop.gradNorm, this->m_stop.xDelta);

        update_solver_info(objFunc.value(x));
		if (solver_info_log)
			std::cout << solver_info << std::endl;

        int f_delta_step_cnt = 0;
        double f_delta = 0;
      
        do
        {
            m_line_search->set_is_final_strategy(m_descent_strategy == m_strategies.size() - 1);

            this->m_current.xDelta = NaN;
            this->m_current.fDelta = NaN;
            this->m_current.gradNorm = NaN;

            //////////// Energy
            double energy;
            {
                POLYSOLVE_SCOPED_STOPWATCH("compute objective function", obj_fun_time, m_logger);
                energy = objFunc.value(x);
            }

            if (!std::isfinite(energy))
            {
                this->m_status = cppoptlib::Status::UserDefined;
                m_error_code = ErrorCode::NAN_ENCOUNTERED;
                log_and_throw_error(m_logger, "[{}][{}] f(x) is nan or inf; stopping", name(), m_line_search->name());
                break;
            }

            f_delta = std::abs(old_energy - energy);
            // stop based on f_delta only if the solver has taken over f_delta_step_tol steps with small f_delta
            this->m_current.fDelta = (f_delta_step_cnt >= f_delta_step_tol) ? f_delta : NaN;

            ///////////// gradient
            {
                POLYSOLVE_SCOPED_STOPWATCH("compute gradient", grad_time, m_logger);
                objFunc.gradient(x, grad);
            }

            const double grad_norm = compute_grad_norm(x, grad);
            if (std::isnan(grad_norm))
            {
                this->m_status = cppoptlib::Status::UserDefined;
                m_error_code = ErrorCode::NAN_ENCOUNTERED;
                log_and_throw_error(m_logger, "[{}][{}] Gradient is nan; stopping", name(), m_line_search->name());
                break;
            }

			{
				POLYSOLVE_SCOPED_STOPWATCH("verify gradient", grad_time, m_logger);
				verify_gradient(objFunc, x, grad);
			}

			if (outfile.is_open())
			{
				outfile << std::setprecision(12) << energy << ", " << grad_norm;
				outfile << "\n";
				outfile.flush();
			}

            this->m_current.gradNorm = grad_norm;
            this->m_status = checkConvergence(this->m_stop, this->m_current);
            if (this->m_status != cppoptlib::Status::Continue)
                break;

            // ------------------------
            // Compute update direction
            // ------------------------
            // Compute a Δx to update the variable
            //
            bool ok = compute_update_direction(objFunc, x, grad, delta_x);

            if (!ok || std::isnan(grad_norm) || (m_strategies[m_descent_strategy]->is_direction_descent() && grad_norm != 0 && delta_x.dot(grad) >= 0))
            {
                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;
                if (m_descent_strategy >= m_strategies.size())
                {
                    this->m_status = cppoptlib::Status::UserDefined;
                    log_and_throw_error(m_logger, "[{}][{}] direction is not a descent direction on last strategy (‖Δx‖={:g}; ‖g‖={:g}; Δx⋅g={:g}≥0); stopping",
                                        name(), m_line_search->name(),
                                        delta_x.norm(), compute_grad_norm(x, grad), delta_x.dot(grad));
                }

                m_logger.debug(
                    "[{}][{}] direction is not a descent direction (‖Δx‖={:g}; ‖g‖={:g}; Δx⋅g={:g}≥0); reverting to {}",
                    name(), m_line_search->name(),
                    delta_x.norm(), compute_grad_norm(x, grad), delta_x.dot(grad), descent_strategy_name());
                this->m_status = cppoptlib::Status::Continue;
                continue;
            }

            const double delta_x_norm = delta_x.norm();
            if (std::isnan(delta_x_norm))
            {
                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;

                if (m_descent_strategy >= m_strategies.size())
                {
                    this->m_status = cppoptlib::Status::UserDefined;
                    log_and_throw_error(m_logger, "[{}][{}] Δx is nan on last strategy; stopping",
                                        name(), m_line_search->name());
                }

                this->m_status = cppoptlib::Status::UserDefined;
                m_logger.debug("[{}][{}] Δx is nan; reverting to {}", name(), m_line_search->name(), descent_strategy_name());
                this->m_status = cppoptlib::Status::Continue;
                continue;
            }

            // Use the maximum absolute displacement value divided by the timestep,
            this->m_current.xDelta = delta_x_norm;
            this->m_status = checkConvergence(this->m_stop, this->m_current);
            if (this->m_status != cppoptlib::Status::Continue)
                break;


			// ---------------
			// Plot energy over descent direction
			// ---------------

			// if (this->m_current.iterations > 8) {
			// 	const double value_ = objFunc.value(x);
			// 	const double rate_ = delta_x.dot(grad);
			// 	std::cout << "descent rate " << rate_ << "\n";
			// 	std::cout << std::setprecision(20) << 0 << " " << value_ << " " << grad.dot(delta_x) << "\n";
			// 	double dt_ = 1e-4;
			// 	while (dt_ < 1e2)
			// 	{
			// 		objFunc.solution_changed(x + delta_x * dt_);
			// 		Eigen::VectorXd grad_;
			// 		objFunc.gradient(x, grad_);
			// 		std::cout << std::setprecision(20) << dt_ << " " << objFunc.value(x + delta_x * dt_) << " " << grad.dot(delta_x) << "\n";
			// 		dt_ *= 1.2;
			// 	}
			// 	exit(0);
			// }
            // ---------------
            // Variable update
            // ---------------

            // Perform a line_search to compute step scale
            double rate = m_line_search->line_search(x, delta_x, objFunc);
            if (std::isnan(rate))
            {
                assert(this->m_status == cppoptlib::Status::Continue);
                if (!m_strategies[m_descent_strategy]->handle_error())
                    ++m_descent_strategy;
                if (m_descent_strategy >= m_strategies.size())
                {
                    this->m_status = cppoptlib::Status::UserDefined; // Line search failed on gradient descent, so quit!
                    log_and_throw_error(m_logger, "[{}][{}] Line search failed on last strategy; stopping", name(), m_line_search->name());
                }

                m_logger.debug("[{}] Line search failed; reverting to {}", name(), descent_strategy_name());
                continue;
            }

            x += rate * delta_x;
            old_energy = energy;

            // Reset this for the next iterations
            // if the strategy got changed, we start counting
            if (m_descent_strategy != previous_strategy)
                current_strategy_iter = 0;
            // if we did enough lower strategy, we revert back to normal
            if (m_descent_strategy != 0 && current_strategy_iter >= m_iter_per_strategy[m_descent_strategy])
            {
                const std::string prev_strategy_name = descent_strategy_name();

                m_descent_strategy = 0;
                for (auto &s : m_strategies)
                    s->reset(x.size());

                m_logger.debug(
                    "[{}][{}] {} was successful for {} iterations; resetting to {}",
                    name(), m_line_search->name(), prev_strategy_name, current_strategy_iter, descent_strategy_name());
            }

            previous_strategy = m_descent_strategy;
            ++current_strategy_iter;

            // -----------
            // Post update
            // -----------
            const double step = (rate * delta_x).norm();

            if (objFunc.stop(x))
            {
                this->m_status = cppoptlib::Status::UserDefined;
                m_error_code = ErrorCode::SUCCESS;
                m_logger.debug("[{}][{}] Objective decided to stop", name(), m_line_search->name());
            }

            objFunc.post_step(this->m_current.iterations, x);

            if (f_delta < this->m_stop.fDelta)
                f_delta_step_cnt++;
            else
                f_delta_step_cnt = 0;

            m_logger.debug(
                "[{}][{}] iter={:d} f={:g} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g} Δx⋅∇f(x)={:g} rate={:g} ‖step‖={:g}"
                " (stopping criteria: max_iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})",
                name(), m_line_search->name(),
                this->m_current.iterations, energy, f_delta,
                this->m_current.gradNorm, this->m_current.xDelta, delta_x.dot(grad), rate, step,
                this->m_stop.iterations, this->m_stop.fDelta, this->m_stop.gradNorm, this->m_stop.xDelta);

            if (++this->m_current.iterations >= this->m_stop.iterations)
                this->m_status = cppoptlib::Status::IterationLimit;

            update_solver_info(energy);
			if (solver_info_log)
				std::cout << solver_info << std::endl;
            objFunc.save_to_file(x);

            // reset the tolerance, since in the first iter it might be smaller
            this->m_stop.gradNorm = g_norm_tol;
        } while (objFunc.callback(this->m_current, x) && (this->m_status == cppoptlib::Status::Continue));

        stop_watch.stop();

        // -----------
        // Log results
        // -----------

        if (!allow_out_of_iterations && this->m_status == cppoptlib::Status::IterationLimit)
            log_and_throw_error(m_logger, "[{}][{}] Reached iteration limit (limit={})", name(), m_line_search->name(), this->m_stop.iterations);
        if (this->m_status == cppoptlib::Status::UserDefined && m_error_code != ErrorCode::SUCCESS)
            log_and_throw_error(m_logger, "[{}][{}] Failed to find minimizer", name(), m_line_search->name());

        double tot_time = stop_watch.getElapsedTimeInSec();
        const bool succeeded = this->m_status == cppoptlib::Status::GradNormTolerance;
        m_logger.log(succeeded ? spdlog::level::info : spdlog::level::err,
            "[{}][{}] Finished: {} Took {:g}s (niters={:d} f={:g} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})"
            " (stopping criteria: max_iters={:d} Δf={:g} ‖∇f‖={:g} ‖Δx‖={:g})",
            name(), m_line_search->name(),
            this->m_status, tot_time, this->m_current.iterations,
            old_energy, f_delta, this->m_current.gradNorm, this->m_current.xDelta,
            this->m_stop.iterations, this->m_stop.fDelta, this->m_stop.gradNorm, this->m_stop.xDelta);

        log_times();
        update_solver_info(objFunc.value(x));
    }

    void Solver::reset(const int ndof)
    {
        this->m_current.reset();
        m_descent_strategy = 0;
        m_error_code = ErrorCode::SUCCESS;

        const std::string line_search_name = solver_info["line_search"];
        solver_info = json();
        solver_info["line_search"] = line_search_name;
        solver_info["iterations"] = 0;

        for (auto &s : m_strategies)
            s->reset(ndof);

        reset_times();
    }

    void Solver::reset_times()
    {
        total_time = 0;
        grad_time = 0;
        line_search_time = 0;
        obj_fun_time = 0;
        constraint_set_update_time = 0;
        if (m_line_search)
        {
            m_line_search->reset_times();
        }
        for (auto &s : m_strategies)
            s->reset_times();
    }

    void Solver::update_solver_info(const double energy)
    {
        solver_info["status"] = this->status();
        solver_info["error_code"] = m_error_code;
        solver_info["energy"] = energy;
        const auto &crit = this->criteria();
        solver_info["iterations"] = crit.iterations;
        solver_info["xDelta"] = crit.xDelta;
        solver_info["fDelta"] = crit.fDelta;
        solver_info["gradNorm"] = crit.gradNorm;
        solver_info["condition"] = crit.condition;

        double per_iteration = crit.iterations ? crit.iterations : 1;

        solver_info["total_time"] = total_time;
        solver_info["time_grad"] = grad_time / per_iteration;
        solver_info["time_line_search"] = line_search_time / per_iteration;
        solver_info["time_constraint_set_update"] = constraint_set_update_time / per_iteration;
        solver_info["time_obj_fun"] = obj_fun_time / per_iteration;

        for (auto &s : m_strategies)
            s->update_solver_info(solver_info, per_iteration);

        if (m_line_search)
        {
            solver_info["line_search_iterations"] = m_line_search->iterations();

            solver_info["time_checking_for_nan_inf"] =
                m_line_search->checking_for_nan_inf_time / per_iteration;
            solver_info["time_broad_phase_ccd"] =
                m_line_search->broad_phase_ccd_time / per_iteration;
            solver_info["time_ccd"] = m_line_search->ccd_time / per_iteration;
            // Remove double counting
            solver_info["time_classical_line_search"] =
                (m_line_search->classical_line_search_time
                 - m_line_search->constraint_set_update_time)
                / per_iteration;
            solver_info["time_line_search_constraint_set_update"] =
                m_line_search->constraint_set_update_time / per_iteration;
        }
    }

    void Solver::log_times()
    {
        m_logger.debug(
            "[{}] grad {:.3g}s, "
            "line_search {:.3g}s, constraint_set_update {:.3g}s, "
            "obj_fun {:.3g}s, checking_for_nan_inf {:.3g}s, "
            "broad_phase_ccd {:.3g}s, ccd {:.3g}s, "
            "classical_line_search {:.3g}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            grad_time, line_search_time,
            constraint_set_update_time + (m_line_search ? m_line_search->constraint_set_update_time : 0),
            obj_fun_time, m_line_search ? m_line_search->checking_for_nan_inf_time : 0,
            m_line_search ? m_line_search->broad_phase_ccd_time : 0, m_line_search ? m_line_search->ccd_time : 0,
            m_line_search ? m_line_search->classical_line_search_time : 0);
    }
} // namespace polysolve::nonlinear
