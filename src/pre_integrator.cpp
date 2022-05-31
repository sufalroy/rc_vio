#include "rcvio/pre_integrator.hpp"

#include <opencv2/core/core.hpp>

#include "rcvio/numerics.h"

namespace rcvio
{
    PreIntegrator::PreIntegrator(const cv::FileStorage &fs_settings)
    {
        sigma_gyro_noise_ = fs_settings["IMU.sigma_g"];
        sigma_gyro_bias_ = fs_settings["IMU.sigma_wg"];
        sigma_accel_noise_ = fs_settings["IMU.sigma_a"];
        sigma_accel_bias_ = fs_settings["IMU.sigma_wa"];

        small_angle_ = fs_settings["IMU.nSmallAngle"];

        gravity_ = fs_settings["IMU.nG"];

        xk1k.setZero(26, 1);
        Pk1k.setZero(24, 24);

        sigma_.setIdentity();
        sigma_.block<3, 3>(0, 0) *= std::pow(sigma_gyro_noise_, 2);
        sigma_.block<3, 3>(3, 3) *= std::pow(sigma_gyro_bias_, 2);
        sigma_.block<3, 3>(6, 6) *= std::pow(sigma_accel_noise_, 2);
        sigma_.block<3, 3>(9, 9) *= std::pow(sigma_accel_bias_, 2);
    }

    void PreIntegrator::propagate(Eigen::VectorXd &xkk,
                                  Eigen::MatrixXd &Pkk,
                                  std::list<ImuData *> &imu_data)
    {
        Eigen::Vector3d gk = xkk.block(7, 0, 3, 1);
        Eigen::Vector4d qk = xkk.block(10, 0, 4, 1);
        Eigen::Vector3d pk = xkk.block(14, 0, 3, 1);
        Eigen::Vector3d vk = xkk.block(17, 0, 3, 1);
        Eigen::Vector3d bg = xkk.block(20, 0, 3, 1);
        Eigen::Vector3d ba = xkk.block(23, 0, 3, 1);

        Eigen::Vector3d gR = gk;

        Eigen::Vector3d vR = vk;

        Eigen::Matrix3d Rk = quatToRot(qk);
        Eigen::Matrix3d Rk_T = Rk.transpose();

        Eigen::Vector3d dp;
        Eigen::Vector3d dv;
        dp.setZero();
        dv.setZero();

        Eigen::Matrix<double, 24, 24> F;
        Eigen::Matrix<double, 24, 24> phi;
        Eigen::Matrix<double, 24, 24> psi;
        F.setZero();
        phi.setZero();
        psi.setIdentity();

        Eigen::Matrix<double, 24, 12> G;
        Eigen::Matrix<double, 24, 24> Q;
        G.setZero();
        Q.setZero();

        Eigen::Matrix3d I;
        I.setIdentity();

        double Dt = 0;

        for (std::list<ImuData *>::const_iterator it = imu_data.begin(); it != imu_data.end(); ++it)
        {
            Eigen::Vector3d wm = (*it)->wm_;
            Eigen::Vector3d am = (*it)->am_;

            double dt = (*it)->dt_;
            Dt += dt;

            Eigen::Vector3d w = wm - bg;
            Eigen::Vector3d a = am - ba;

            bool is_small_angle = false;
            if (w.norm() < small_angle_)
            {
                is_small_angle = true;
            }

            double w1 = w.norm();
            double wdt = w1 * dt;
            double wdt2 = wdt * wdt;
            double coswdt = std::cos(wdt);
            double sinwdt = std::sin(wdt);
            Eigen::Matrix3d wx = skew(w);
            Eigen::Matrix3d wx2 = wx * wx;
            Eigen::Matrix3d vx = skew(vk);

            F.block<3, 3>(9, 9) = -wx;
            F.block<3, 3>(9, 18) = -I;
            F.block<3, 3>(12, 9) = -Rk_T * vx;
            F.block<3, 3>(12, 15) = Rk_T;
            F.block<3, 3>(15, 6) = -gravity_ * Rk;
            F.block<3, 3>(15, 9) = -gravity_ * skew(gk);
            F.block<3, 3>(15, 15) = -wx;
            F.block<3, 3>(15, 18) = -vx;
            F.block<3, 3>(15, 21) = -I;
            phi = Eigen::Matrix<double, 24, 24>::Identity() + dt * F;
            psi = phi * psi;

            G.block<3, 3>(9, 0) = -I;
            G.block<3, 3>(15, 0) = -vx;
            G.block<3, 3>(15, 6) = -I;
            G.block<3, 3>(18, 3) = I;
            G.block<3, 3>(21, 9) = I;
            Q = dt * G * sigma_ * (G.transpose());

            Pkk.block(0, 0, 24, 24) = phi * (Pkk.block(0, 0, 24, 24)) * (phi.transpose()) + Q;

            Eigen::Matrix3d delta_R;
            double f1, f2, f3, f4;
            if (is_small_angle)
            {
                delta_R = I - dt * wx + (std::pow(dt, 2) / 2) * wx2;
                assert(std::isnan(delta_R.norm()) != true);

                f1 = -std::pow(dt, 3) / 3;
                f2 = std::pow(dt, 4) / 8;
                f3 = -std::pow(dt, 2) / 2;
                f4 = std::pow(dt, 3) / 6;
            }
            else
            {
                delta_R = I - (sinwdt / w1) * wx + ((1 - coswdt) / std::pow(w1, 2)) * wx2;
                assert(std::isnan(delta_R.norm()) != true);

                f1 = (wdt * coswdt - sinwdt) / std::pow(w1, 3);
                f2 = 0.5 * (wdt2 - 2 * coswdt - 2 * wdt * sinwdt + 2) / std::pow(w1, 4);
                f3 = (coswdt - 1) / std::pow(w1, 2);
                f4 = (wdt - sinwdt) / std::pow(w1, 3);
            }

            Rk = delta_R * Rk;
            Rk_T = Rk.transpose();

            dp += dv * dt;
            dp += Rk_T * (0.5 * std::pow(dt, 2) * I + f1 * wx + f2 * wx2) * a;
            dv += Rk_T * (dt * I + f3 * wx + f4 * wx2) * a;

            pk = vR * Dt - 0.5 * gravity_ * gR * std::pow(Dt, 2) + dp;
            vk = Rk * (vR - gravity_ * gR * Dt + dv);
            gk = Rk * gR;
            gk.normalize();
        }

        xk1k = xkk;
        xk1k.block(10, 0, 4, 1) = rotToQuat(Rk);
        xk1k.block(14, 0, 3, 1) = pk;
        xk1k.block(17, 0, 3, 1) = vk;

        int clone_states = (xkk.rows() - 26) / 7;
        if (clone_states > 0)
        {
            Pkk.block(0, 24, 24, 6 * clone_states) = psi * Pkk.block(0, 24, 24, 6 * clone_states);
            Pkk.block(24, 0, 6 * clone_states, 24) = Pkk.block(0, 24, 24, 6 * clone_states).transpose();
        }

        Pkk = 0.5 * (Pkk + Pkk.transpose());
        Pk1k = Pkk;
    }
}