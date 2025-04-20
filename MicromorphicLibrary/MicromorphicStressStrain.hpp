#ifndef MICROMORPHIC_STRESS_STRAIN_HPP
#define MICROMORPHIC_STRESS_STRAIN_HPP

#include "mfem.hpp"

/**
 * Compute the elemental strain eps from the displacement field u.
 */
void CalcStrain(const mfem::GridFunction &u, mfem::GridFunction &eps);

/**
 * Compute the elemental couple strain e from the displacement field u and micro displacement
 * gradient field phi.
 */
void CalcCoupleStrain(const mfem::GridFunction &u, const mfem::GridFunction &phi, mfem::GridFunction &e);

/**
 * Compute the elemental micro strain K from the micro displacement gradient field K.
 */
void CalcMicroStrain(const mfem::GridFunction &phi, mfem::GridFunction &K);

/**
 * Compute the elemental Cauchy stress sigma from the elemental strain eps and 
 * the elemental couple strain e.
 */
void CalcStress(mfem::GridFunction const &eps, mfem::GridFunction const &e, mfem::Coefficient &mu, 
                mfem::Coefficient &lambda, mfem::Coefficient &c1, mfem::Coefficient &c2, 
                mfem::GridFunction &sigma);

/**
 * Compute the elemental couple stress s from the elemental strain eps and the 
 * elemental couple strain e.
 */
void CalcCoupleStress(const mfem::GridFunction &eps, const mfem::GridFunction &e, mfem::Coefficient &b1, 
                        mfem::Coefficient &b2, mfem::Coefficient &c1, mfem::Coefficient &c2, 
                        mfem::GridFunction &s);

/**
 * Compute the elemental microstress S from the elemental microstrain K.
 */
void CalcMicroStress(const mfem::GridFunction &K, mfem::Coefficient &A1, mfem::Coefficient &A2, 
                        mfem::GridFunction &S);

#endif
