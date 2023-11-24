#include "BoxConstraintSolver.hpp"

#include "descent_strategies/box_constraints/LBFGSB.hpp"
#include "descent_strategies/box_constraints/MMA.hpp"

#include <jse/jse.h>
#include <polysolve/JSONUtils.hpp>

#include <fstream>

namespace nlohmann
{
	template <typename T, int nrows, int ncols, int maxdim1, int maxdim2>
	struct adl_serializer<Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2>>
	{
		static void to_json(json &j, const Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2> &matrix)
		{
			for (int row = 0; row < matrix.rows(); ++row)
			{
				json column = json::array();
				for (int col = 0; col < matrix.cols(); ++col)
				{
					column.push_back(matrix(row, col));
				}
				j.push_back(column);
			}
		}

		static void from_json(const json &j, Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2> &matrix)
		{
			using Scalar = typename Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2>::Scalar;
			assert(j.size() > 0);
			assert(nrows >= j.size() || nrows == -1);
			assert(j.at(0).is_number() || ncols == -1 || ncols >= j.at(0).size());
			
			const int n_cols = j.at(0).is_number() ? 1 : j.at(0).size();
			if (nrows == -1 || ncols == -1)
				matrix.setZero(j.size(), n_cols);
			
			for (std::size_t row = 0; row < j.size(); ++row)
			{
				const auto& jrow = j.at(row);
				if (jrow.is_number())
					matrix(row) = jrow;
				else
					for (std::size_t col = 0; col < jrow.size(); ++col)
					{
						const auto& value = jrow.at(col);
						matrix(row, col) = value.get<Scalar>();
					}
			}
		}
	};
} // namespace nlohmann

namespace polysolve::nonlinear
{

    // Static constructor
    std::unique_ptr<Solver> BoxConstraintSolver::create(
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

        auto solver = std::make_unique<BoxConstraintSolver>(solver_name, solver_params, characteristic_length, logger);

        if (solver_name == "LBFGSB" || solver_name == "L-BFGS-B")
        {
            solver->add_strategy(std::make_unique<LBFGSB>(
                solver_params, characteristic_length, logger));
        }
        else if (solver_name == "MMA")
        {
            if (solver->line_search()->name() != "None")
                log_and_throw_error(logger, "Invalid linesearch for MMA; MMA requires 'None' linesearch, instead got {}", solver->line_search()->name());

            solver->add_strategy(std::make_unique<MMA>(
                solver_params, characteristic_length, logger));
        }
        else
            throw std::runtime_error("Unrecognized solver type: " + solver_name);

        solver->set_strategies_iterations(solver_params);
        return solver;
    }

    std::vector<std::string> BoxConstraintSolver::available_solvers()
    {
        return {"L-BFGS-B",
                "MMA"};
    }

    BoxConstraintSolver::BoxConstraintSolver(const std::string &name,
                                             const json &solver_params,
                                             const double characteristic_length,
                                             spdlog::logger &logger)
        : Superclass(name, solver_params, characteristic_length, logger)
    {
        json box_constraint_params = solver_params["box_constraints"];
        if (box_constraint_params["max_change"].is_number())
            max_change_val_ = box_constraint_params["max_change"];
        else
            max_change_ = box_constraint_params["max_change"];

        if (box_constraint_params.contains("bounds") && box_constraint_params["bounds"].is_array() && box_constraint_params["bounds"].size() == 2)
        {
            bounds_ = box_constraint_params["bounds"];
        }
    }

    bool BoxConstraintSolver::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        const TVector lower_bound = get_lower_bound(x);
        const TVector upper_bound = get_upper_bound(x);

        return m_strategies[m_descent_strategy]->compute_boxed_update_direction(
            objFunc, x, grad,
            lower_bound, upper_bound,
            direction);
    }

    double BoxConstraintSolver::compute_grad_norm(const Eigen::VectorXd &x,
                                                  const Eigen::VectorXd &grad) const
    {
        auto min = get_lower_bound(x, false);
        auto max = get_upper_bound(x, false);

        return ((x - grad).cwiseMax(min).cwiseMin(max) - x).norm();
        // Eigen::VectorXd proj_grad = grad;
        // for (int i = 0; i < x.size(); i++)
        // 	if (x(i) < min(i) + 1e-14 || x(i) > max(i) - 1e-14)
        // 		proj_grad(i) = 0;

        // return proj_grad.norm();
    }

    Eigen::VectorXd BoxConstraintSolver::get_lower_bound(const Eigen::VectorXd &x,
                                                         bool consider_max_change) const
    {
        Eigen::VectorXd min;
        if (bounds_.cols() == x.size())
            min = bounds_.row(0);
        else if (bounds_.size() == 2)
            min = Eigen::VectorXd::Constant(x.size(), 1, bounds_(0));
        else
            log_and_throw_error(m_logger, "Invalid bounds!");

        if (consider_max_change)
            return min.array().max(x.array() - get_max_change(x).array());
        else
            return min;
    }

    Eigen::VectorXd BoxConstraintSolver::get_upper_bound(const Eigen::VectorXd &x,
                                                         bool consider_max_change) const
    {
        Eigen::VectorXd max;
        if (bounds_.cols() == x.size())
            max = bounds_.row(1);
        else if (bounds_.size() == 2)
            max = Eigen::VectorXd::Constant(x.size(), 1, bounds_(1));
        else
            log_and_throw_error(m_logger, "Invalid bounds!");

        if (consider_max_change)
            return max.array().min(x.array() + get_max_change(x).array());
        else
            return max;
    }

    Eigen::VectorXd BoxConstraintSolver::get_max_change(const Eigen::VectorXd &x) const
    {
        if (max_change_.size() == x.size())
            return max_change_;
        else if (max_change_val_ > 0)
            return Eigen::VectorXd::Ones(x.size()) * max_change_val_;
        else
            log_and_throw_error(m_logger, "Invalid max change!");

        return Eigen::VectorXd();
    }

} // namespace polysolve::nonlinear