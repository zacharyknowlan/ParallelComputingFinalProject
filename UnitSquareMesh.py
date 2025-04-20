import gmsh

# Parameters
MeshFilename = "/lore/knowlz/ParallelComputingProject/UnitSquare.msh"
SideLength = 1.
MeshSize = SideLength/100.

gmsh.initialize()

# Create the beam and its edge midpoints
Square = gmsh.model.occ.addRectangle(0., 0., 0., SideLength, SideLength)
gmsh.model.occ.synchronize()

# Set mesh size at all mesh points
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), MeshSize)

# Make the mesh quadrilateral
gmsh.model.mesh.setTransfiniteSurface(Square)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 6)

# Create physical groups for the edges
Boundary = gmsh.model.getBoundary([(2, Square)], oriented=False)
gmsh.model.addPhysicalGroup(1, [Boundary[0][1]], tag=1, name="Bottom")
gmsh.model.addPhysicalGroup(1, [Boundary[1][1]], tag=2, name="Right")
gmsh.model.addPhysicalGroup(1, [Boundary[2][1]], tag=3, name="Top")
gmsh.model.addPhysicalGroup(1, [Boundary[3][1]], tag=4, name="Left")

# Create physical group for the surface
Surface = [Entity[1] for Entity in gmsh.model.getEntities(dim=2)]
gmsh.model.addPhysicalGroup(2, Surface, tag=6, name="Surface")

gmsh.model.mesh.generate(dim=2)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(MeshFilename)

gmsh.finalize()