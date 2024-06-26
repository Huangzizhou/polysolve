// ADAM from "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"

#include "ADAM.hpp"

#include <polysolve/Utils.hpp>

namespace polysolve::nonlinear
{

    ADAM::ADAM(const json &solver_params,
               const bool is_stochastic,
               const double characteristic_length,
               spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger), is_stochastic_(is_stochastic)
    {
        std::string param_name = is_stochastic ? "StochasticADAM" : "ADAM";
        alpha_ = extract_param(param_name, "alpha", solver_params);
        beta_1_ = extract_param(param_name, "beta_1", solver_params);
        beta_2_ = extract_param(param_name, "beta_2", solver_params);
        epsilon_ = extract_param(param_name, "epsilon", solver_params);
        if (is_stochastic)
            erase_component_probability_ = extract_param("StochasticADAM", "erase_component_probability", solver_params);
    }

    void ADAM::reset(const int ndof)
    {
        Superclass::reset(ndof);
        m_prev_ = Eigen::VectorXd::Zero(ndof);
        v_prev_ = Eigen::VectorXd::Zero(ndof);
        t_ = 0;
    }

    bool ADAM::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        if (m_prev_.size() == 0)
            m_prev_ = Eigen::VectorXd::Zero(x.size());
        if (v_prev_.size() == 0)
            v_prev_ = Eigen::VectorXd::Zero(x.size());

        TVector grad_modified = grad;

        if (is_stochastic_)
        {
            Eigen::VectorXd mask = (Eigen::VectorXd::Random(direction.size()).array() + 1.) / 2.;
            grad_modified = (mask.array() < erase_component_probability_).select(Eigen::VectorXd::Zero(grad_modified.size()), grad_modified);
        }

        TVector m = beta_1_ * m_prev_ + (1 - beta_1_) * grad_modified;
        TVector v = beta_2_ * v_prev_.array() + (1 - beta_2_) * grad_modified.array().square();

        m /= (1 - pow(beta_1_, t_));
        v /= (1 - pow(beta_2_, t_));

        direction = -alpha_ * m.array() / (v.array() + epsilon_).sqrt();

        ++t_;

        return true;
    }
} // namespace polysolve::nonlinear
