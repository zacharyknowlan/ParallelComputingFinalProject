#include "GlobalStaticCondensation.hpp"

GlobalStaticCondensation::GlobalStaticCondensation(mfem::SparseMatrix* const global_mat, mfem::Vector* const sol_vec, 
                                                    mfem::Vector* const rhs_vec)
{
    MFEM_VERIFY(global_mat->Width() == sol_vec->Size(), "Matrix width and solution vector size must be the same.")
    MFEM_VERIFY(sol_vec->Size() == rhs_vec->Size(), "Solution and RHS vector must be of same size.")
    
    A = global_mat;
    x = sol_vec;
    b = rhs_vec;

    bc_marker_list.SetSize(x->Size());
    bc_count.SetSize(x->Size());
}

void GlobalStaticCondensation::PopulateBCMarkerList(const mfem::Array<int>& block_offsets, 
                                                    const mfem::Array<int>& bc_vdofs_1, 
                                                    const mfem::Array<int>& bc_vdofs_2)
{   
    bc_marker_list = 0;
    bc_count = 0;

    for (int ii=0; ii<bc_vdofs_1.Size(); ii++)
    {
        bc_marker_list[bc_vdofs_1[ii]] = 1;
    }    
    for(int jj=0; jj<bc_vdofs_2.Size(); jj++)
    {
        bc_marker_list[bc_vdofs_2[jj]+block_offsets[1]] = 1;
    }

    // Define condensed system size based on BCs
    CondensedSystemSize = bc_marker_list.Size() - bc_marker_list.Sum();

    bc_count[0] = bc_marker_list[0];

    for(int kk=1; kk<bc_marker_list.Size(); kk++)
    {
        bc_count[kk] = bc_count[kk-1] + bc_marker_list[kk];
    }
}

void GlobalStaticCondensation::CreateCondensedSystem()
{
    A_cond = mfem::SparseMatrix(CondensedSystemSize);
    A_cond = 0.;
    x_cond = mfem::Vector(CondensedSystemSize);
    b_cond = mfem::Vector(CondensedSystemSize);
    b_cond = 0.;

    int cond_row = 0;
    int cond_col = 0;

    const int *I = A->GetI();
    const int *J = A->GetJ();
    const mfem::real_t *data = A->GetData();

    for (int ii=0; ii<A->Height(); ii++)
    {
        if (bc_marker_list[ii] == 0)
        {
            for (int jj=I[ii]; jj<I[ii+1]; jj++)
            {
                if (bc_marker_list[J[jj]] == 0)
                {
                    cond_col = J[jj] - bc_count[J[jj]];
                    A_cond.Set(cond_row, cond_col, data[jj]); // assemble stiffness matrix
                }
                else if (bc_marker_list[J[jj]] == 1)
                {
                    b_cond[cond_row] -= data[jj] * (*x)[J[jj]]; // subtract essential BCs
                }
            }
            b_cond[cond_row] += (*b)[ii]; // add existing rhs terms
            cond_row++;
        }
    }
    A_cond.Finalize();
    A_cond.SortColumnIndices();
}

void GlobalStaticCondensation::RegularizeMatrix(double perturbation)
{
    const int *I = A_cond.GetI();
    const int *J = A_cond.GetJ();
    mfem::real_t *data = A_cond.GetData();

    for (int ii=0; ii<A_cond.Height(); ii++)
    {
        for (int jj=I[ii]; jj<I[ii+1]; jj++)
        {
            if (ii == J[jj]) 
            {
                data[jj] += perturbation;
                break;
            }
        }
    }
}

void GlobalStaticCondensation::RecoverGlobalSolution()
{
    int jj = 0;
    for (int ii=0; ii<x->Size(); ii++)
    {
        if (bc_marker_list[ii] == 0)
        {
            (*x)[ii] = x_cond[jj];
            jj++;
        }
    }
}

void GlobalStaticCondensation::AddConstraintSystem(mfem::SparseMatrix* const ConstraintMat, 
                                                        mfem::Vector* const ConstraintRHSVec)
{
    B = ConstraintMat;
    B_cond = mfem::SparseMatrix(B->Height(), CondensedSystemSize);
    B_cond = 0.;
    
    r = ConstraintRHSVec;
    r_cond.SetSize(B->Height());
    r_cond = 0.;
}

void GlobalStaticCondensation::CondenseConstraintSystem()
{
    int cond_row = 0;
    int cond_col = 0;

    const int *I = B->GetI();
    const int *J = B->GetJ();
    const mfem::real_t *data = B->GetData();

    for (int ii=0; ii<B->Height(); ii++)
    {
        for (int jj=I[ii]; jj<I[ii+1]; jj++)
        {
            if (bc_marker_list[J[jj]] == 0)
            {
                cond_col = J[jj] - bc_count[J[jj]];
                B_cond.Set(cond_row, cond_col, data[jj]); // assemble stiffness matrix
            }
            else if (bc_marker_list[J[jj]] == 1)
            {
                r_cond[cond_row] -= data[jj] * (*x)[J[jj]]; // subtract essential BCs
            }
        }
        cond_row++;
    }
    B_cond.Finalize();
}
