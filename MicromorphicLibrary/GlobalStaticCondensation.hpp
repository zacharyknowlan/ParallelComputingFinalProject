#ifndef GLOBALSTATICCOND_HPP
#define GLOBALSTATICCOND_HPP

#include "mfem.hpp"

class GlobalStaticCondensation
{
    private:

        // Global Linear system
        mfem::SparseMatrix* A; // Not owned
        mfem::SparseMatrix A_cond;
        mfem::Vector* x; // Not owned
        mfem::Vector x_cond;
        mfem::Vector* b; // Not owned
        mfem::Vector b_cond;

        // Constraint system
        mfem::SparseMatrix* B; // Not owned
        mfem::SparseMatrix B_cond;
        mfem::Vector* r; // Not owned
        mfem::Vector r_cond;

        // Essential VDof information
        mfem::Array<int> bc_marker_list;
        mfem::Array<int> bc_count;
        int CondensedSystemSize;

    public:

        GlobalStaticCondensation() {}

        // Change to "AddGlobalSystem"
        GlobalStaticCondensation(mfem::SparseMatrix* const global_mat, mfem::Vector* const sol_vec, 
                                    mfem::Vector* const rhs_vec);
        
        // Change to "AddEssentialVDofs"
        void PopulateBCMarkerList(const mfem::Array<int>& block_offsets, const mfem::Array<int>& bc_vdofs_1, 
                                    const mfem::Array<int>& bc_vdofs_2);

        // Change to "CondenseGlobalSystem"
        void CreateCondensedSystem();

        [[nodiscard]] mfem::SparseMatrix& StiffnessMatrix() noexcept {return A_cond;}

        [[nodiscard]] mfem::Vector& SolutionVector() noexcept {return x_cond;}

        [[nodiscard]] mfem::Vector& RHSVector() noexcept {return b_cond;}

        [[nodiscard]] const mfem::Array<int>& GetBCMarkerList() const {return bc_marker_list;}

        void RegularizeMatrix(double perturbation);

        void RecoverGlobalSolution();

        void AddConstraintSystem(mfem::SparseMatrix* const ConstraintMat, mfem::Vector* const ConstraintRHSVec);

        void CondenseConstraintSystem();

        [[nodiscard]] mfem::SparseMatrix& ConstraintMatrix() noexcept {return B_cond;}

        [[nodiscard]] mfem::Vector& RHSConstraintVector() noexcept {return r_cond;}

        ~GlobalStaticCondensation() {}
};

#endif 
