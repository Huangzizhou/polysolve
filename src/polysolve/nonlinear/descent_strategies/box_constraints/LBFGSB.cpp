#include "LBFGSB.hpp"

#include <LBFGSpp/Cauchy.h>
#include <LBFGSpp/SubspaceMin.h>

#include <spdlog/fmt/ostr.h>
// #include <spdlog/fmt/bundled/format.h>

namespace {
    template <typename Derived>
    void print_matrix(const Eigen::MatrixBase<Derived> &x)
    {
        std::cout << std::setprecision(16);
        for (int i = 0; i < x.size(); i++)
            std::cout << x(i) << " ";
        std::cout << "\n";
    }
}

namespace polysolve::nonlinear
{
    LBFGSB::LBFGSB(const json &solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger)
    {
        m_history_size = solver_params["LBFGSB"]["history_size"];
        print_direction = solver_params["LBFGSB"]["print_direction"];

        if (m_history_size <= 0)
            log_and_throw_error(logger, "LBFGSB history_size must be >=1, instead got {}", m_history_size);
    }

    void LBFGSB::reset(const int ndof)
    {
        Superclass::reset(ndof);

        reset_history(ndof);
    }

    void LBFGSB::reset_history(const int ndof)
    {
        m_bfgs.reset(ndof, m_history_size);
        m_prev_x.resize(0);
        m_prev_grad.resize(ndof);
    }

    bool LBFGSB::compute_boxed_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        const TVector &lower_bound,
        const TVector &upper_bound,
        TVector &direction)
    {
        TVector cauchy_point(x.size()), vecc;
        std::vector<int> newact_set, fv_set;
        if (m_prev_x.size() == 0)
        {
            // Use gradient descent in the first iteration or if the previous iteration failed
            // direction = -grad;

            LBFGSpp::Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, grad, lower_bound, upper_bound, cauchy_point, vecc, newact_set, fv_set);

            direction = cauchy_point - x;
        }
        else
        {
            // Update s and y
            // s_{i+1} = x_{i+1} - x_i
            // y_{i+1} = g_{i+1} - g_i
            if ((x - m_prev_x).dot(grad - m_prev_grad) > 1e-9 * (grad - m_prev_grad).squaredNorm())
                m_bfgs.add_correction(x - m_prev_x, grad - m_prev_grad);

            // Recursive formula to compute d = -H * g
            // m_bfgs.apply_Hv(grad, -Scalar(1), direction);

            LBFGSpp::Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, grad, lower_bound, upper_bound, cauchy_point, vecc, newact_set, fv_set);

            LBFGSpp::SubspaceMin<Scalar>::subspace_minimize(m_bfgs, x, cauchy_point, grad, lower_bound, upper_bound,
                                                            vecc, newact_set, fv_set, /*Maximum number of iterations*/ max_submin, direction);
        }

        if (print_direction)
        {
            // m_logger.error("x: {}", x.transpose());
            // m_logger.error("grad: {}", grad.transpose());
            // m_logger.error("direc: {}", direction.transpose());
            std::cout << "x: ";
            print_matrix(x);
            std::cout << "grad: ";
            print_matrix(grad);
            std::cout << "direc: ";
            print_matrix(direction);
        }

        m_prev_x = x;
        m_prev_grad = grad;

        return true;
    }
} // namespace polysolve::nonlinear
