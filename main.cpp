#include <fstream>
#include <string>
#include "mfem.hpp"
#include "MicromorphicIntegrator.hpp"
 
int main(int argc, char** argv)
{   
    // Run parameters
    std::string MeshFile = "../UnitSquare.msh";
    std::string ResultFilename = "../result.vtk";
    double Mu = 454545.45454545453;
    double Lambda = 113636.36363636363;
    double ls = 0.1;

    // Tell MFEM about the device
    //mfem::Device device("cpu");
    mfem::Device device("cuda");
    device.Print();

    // Create mesh from file and finite element spaces 
    auto mesh = mfem::Mesh(MeshFile.c_str(), 1, 1);
    int dim = mesh.SpaceDimension();

    auto u_ec = mfem::H1_FECollection(3, dim, mfem::BasisType::GaussLobatto);
    auto u_space = mfem::FiniteElementSpace(&mesh, &u_ec, dim);

    auto phi_ec = mfem::H1_FECollection(2, dim, mfem::BasisType::GaussLobatto);
    auto phi_space = mfem::FiniteElementSpace(&mesh, &phi_ec, dim*dim);

    // Define material properties
    auto MuAlpha = mfem::ConstantCoefficient(2.*Mu);
    auto LambdaAlpha = mfem::ConstantCoefficient(2.*Lambda);
    auto MuBeta = mfem::ConstantCoefficient(Mu);
    auto LambdaBeta = mfem::ConstantCoefficient(Lambda);
    auto MuGamma = mfem::ConstantCoefficient(Mu);
    auto LambdaGamma = mfem::ConstantCoefficient(Lambda);
    auto MuA = mfem::ConstantCoefficient(ls*ls*Mu);
    auto LambdaA = mfem::ConstantCoefficient(ls*ls*Lambda);
    
    mfem::Array<int> block_offsets(3); 
    block_offsets[0] = 0;
    block_offsets[1] = u_space.GetVSize();
    block_offsets[2] = phi_space.GetVSize();
    block_offsets.PartialSum();

    // Initialize solution, RHS, and aux vectors
    auto x = mfem::BlockVector(block_offsets, device.GetMemoryType());
    auto b = mfem::BlockVector(block_offsets, device.GetMemoryType());
    auto negative_x = mfem::BlockVector(block_offsets, device.GetMemoryType());
    
    // Set device config and use 
    if (device.IsEnabled())
    {
        x.GetBlock(0).UseDevice(true);
        x.GetBlock(1).UseDevice(true);
        x.UseDevice(true);
        x.GetBlock(0).HostWrite();
        x.GetBlock(1).HostWrite();
        x.HostWrite();
        b.GetBlock(0).UseDevice(true);
        b.GetBlock(1).UseDevice(true);
        b.UseDevice(true);
        b.GetBlock(0).HostWrite();
        b.GetBlock(1).HostWrite();
        b.HostWrite();
        negative_x.GetBlock(0).UseDevice(true);
        negative_x.GetBlock(1).UseDevice(true);
        negative_x.UseDevice(true);
        negative_x.GetBlock(0).HostWrite();
        negative_x.GetBlock(1).HostWrite();
        negative_x.HostWrite();
    }
    
    x.GetBlock(0) = 0.;
    x.GetBlock(1) = 0.;
    x.SyncFromBlocks();
    x.SyncToBlocks();
    b.GetBlock(0) = 0.;
    b.GetBlock(1) = 0.;
    b.SyncFromBlocks();
    b.SyncToBlocks();
    negative_x.GetBlock(0) = 0.;
    negative_x.GetBlock(1) = 0.;
    negative_x.SyncFromBlocks();
    negative_x.SyncToBlocks();

    // Arrays for boundary conditions
    mfem::Array<int> uBCDOFs, phiBCDOFs, phi0BCDOFs;
    mfem::Array<int> LeftEdge({0, 0, 0, 1, 0});

    // Populate solution vector with non-zero essential boundary condition values
    phi_space.GetEssentialTrueDofs(LeftEdge, phi0BCDOFs, 0);
    for (int ii=0; ii<phi0BCDOFs.Size(); ii++)
    {
        x.GetBlock(1)[phi0BCDOFs[ii]] = 0.001;
        negative_x.GetBlock(1)[phi0BCDOFs[ii]] = -0.001;
    }

    // Create a global array of essential boundary condition dofs
    u_space.GetEssentialTrueDofs(LeftEdge, uBCDOFs);
    phi_space.GetEssentialTrueDofs(LeftEdge, phiBCDOFs);

    mfem::Array<int> GlobalBCDOFs(uBCDOFs.Size() + phiBCDOFs.Size(), device.GetMemoryType());

    if (device.IsEnabled()) {GlobalBCDOFs.HostReadWrite();}
    int offset = uBCDOFs.Size();
    int u_size = u_space.GetVSize();
    for (int ii=0; ii<uBCDOFs.Size(); ii++) {GlobalBCDOFs[ii] = uBCDOFs[ii];}
    for (int ii=0; ii<phiBCDOFs.Size(); ii++) {GlobalBCDOFs[ii + offset] = u_size + phiBCDOFs[ii];}

    // Right edge force boundary condition
    auto xForceComponent = mfem::Vector({0., 1e3, 0., 0., 0.});
    auto BoundaryForce = mfem::VectorArrayCoefficient(mesh.SpaceDimension());
    BoundaryForce.Set(0, new mfem::PWConstCoefficient(xForceComponent));
    BoundaryForce.Set(1, new mfem::ConstantCoefficient(0.));

    // Create global block operator representation
    auto a1 = mfem::BilinearForm(&u_space);
    a1.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaAlpha, &MuAlpha, Block::TopLeft)); 
    a1.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
    a1.Assemble();

    auto a2 = mfem::MixedBilinearForm(&phi_space, &u_space);
    a2.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaBeta, &MuBeta, Block::TopRight)); 
    a2.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
    a2.Assemble();

    auto a3 = mfem::MixedBilinearForm(&u_space, &phi_space);
    a3.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaBeta, &MuBeta, Block::BottomLeft));
    a3.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
    a3.Assemble();

    auto a4 = mfem::BilinearForm(&phi_space);
    a4.AddDomainIntegrator(new MicromorphicIntegrator(&LambdaGamma, &MuGamma, &LambdaA, &MuA, Block::BottomRight));
    a4.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
    a4.Assemble();

    if (device.IsEnabled())
    {
        // Make sure negative_x and b are on the device
        negative_x.GetBlock(0).Read();
        negative_x.GetBlock(1).Read();
        negative_x.Read();
        b.GetBlock(0).Write();
        b.GetBlock(1).Write();
        b.Write();
    }

    // Subtract non-zero essential phi boundary conditions (on the device)
    a2.Mult(negative_x.GetBlock(1), b.GetBlock(0));
    a4.Mult(negative_x.GetBlock(1), b.GetBlock(1));
    
    if (device.IsEnabled())
    {
        // Copy the modified RHS Vector back to the host
        b.GetBlock(0).HostWrite();
        b.GetBlock(1).HostWrite();
        b.HostWrite();
    }
    
    // Assemble force boundary condition on the host and add contributions
    auto b1 = mfem::LinearForm(&u_space);
    b1.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(BoundaryForce));
    b1.Assemble();
    for (int ii=0; ii<u_space.GetVSize(); ii++) {b.GetBlock(0)[ii] += b1[ii];}

    // Set RHS values back to solution values to work with A_constrained
    for (int ii=0; ii<uBCDOFs.Size(); ii++) {b.GetBlock(0)[uBCDOFs[ii]] = 0.;}
    for (int ii=0; ii<phiBCDOFs.Size(); ii++) {b.GetBlock(1)[phiBCDOFs[ii]] = 0.;}
    for (int ii=0; ii<phi0BCDOFs.Size(); ii++) {b.GetBlock(1)[phi0BCDOFs[ii]] = 0.001;}

    if (device.IsEnabled())
    {
        // Make sure both vectors end up on the device
        x.GetBlock(0).Read();
        x.GetBlock(1).Read();
        x.Read();
        b.GetBlock(0).Read();
        b.GetBlock(1).Read();
        b.Read();
    }
    
    // Create a constrained representation of the block operator
    mfem::ConstrainedOperator a1_constrained(&a1, uBCDOFs);
    mfem::RectangularConstrainedOperator a2_constrained(&a2, phiBCDOFs, uBCDOFs);
    mfem::RectangularConstrainedOperator a3_constrained(&a3, uBCDOFs, phiBCDOFs);
    mfem::ConstrainedOperator a4_constrained(&a4, phiBCDOFs);
    
    auto A_constrained = mfem::BlockOperator(block_offsets);
    A_constrained.SetBlock(0, 0, &a1_constrained);
    A_constrained.SetBlock(0, 1, &a2_constrained);
    A_constrained.SetBlock(1, 0, &a3_constrained);
    A_constrained.SetBlock(1, 1, &a4_constrained);

    // Assemble the global diagonal for the preconditioner
    auto a1_diag = mfem::Vector(u_space.GetVSize());
    auto a4_diag = mfem::Vector(phi_space.GetVSize());
    auto GlobalDiag = mfem::Vector(block_offsets[2]);
    
    if (device.IsEnabled())
    {
        a1_diag.UseDevice(true);
        a4_diag.UseDevice(true);
        GlobalDiag.UseDevice(true);
        GlobalDiag.HostReadWrite();
    }
    
    a1.AssembleDiagonal(a1_diag);
    a4.AssembleDiagonal(a4_diag);

    if (device.IsEnabled())
    { 
        a1_diag.HostRead();
        a4_diag.HostRead();
    }

    for (int ii=0; ii<a1_diag.Size(); ii++) {GlobalDiag[ii] = a1_diag[ii];}
    for (int ii=0; ii<a4_diag.Size(); ii++) {GlobalDiag[ii+block_offsets[1]] = a4_diag[ii];}

    // Solve the linear system
    auto preconditioner = mfem::OperatorJacobiSmoother(GlobalDiag, GlobalBCDOFs, 1.0);
    auto solver = mfem::GMRESSolver();
    solver.SetOperator(A_constrained);
    solver.SetKDim(100);
    solver.SetPreconditioner(preconditioner);
    solver.iterative_mode = false;
    solver.SetRelTol(1e-4);
    solver.SetAbsTol(1e-8);
    solver.SetMaxIter(6000);
    solver.SetPrintLevel(1);
    solver.Mult(b, x);

    if (device.IsEnabled())
    {
        // Copy the solution vector to the device.
        x.GetBlock(0).HostRead();
        x.GetBlock(1).HostRead();
        x.HostRead();
    }

    // Create gridfunctions of the solution
    mfem::GridFunction u, phi;
    u.MakeRef(&u_space, x.GetBlock(0), 0);
    phi.MakeRef(&phi_space, x.GetBlock(1), 0);

    // Save result
    std::ofstream file(ResultFilename);
    file.precision(16);
    mesh.PrintVTK(file, 0);
    u.SaveVTK(file, "u", 0);
    phi.SaveVTK(file, "phi", 0);
    file.close();
    return 0;
}
