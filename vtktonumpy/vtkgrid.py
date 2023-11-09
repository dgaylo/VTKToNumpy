"""
Contains VTKGrid, a wrapper for VTK vtkRectilinearGrid
"""

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
from vtkmodules.util.numpy_support import vtk_to_numpy


class VTKGrid:
    """A wrapper for VTK vtkRectilinearGrid which makes adjustments for cell
    data focused workflows and outputs numpy arrays"""

    @staticmethod
    def __boundsToCellCenter(bounds: np.ndarray) -> np.ndarray:
        """
        Convert cell bounds to cell center coordinates
        """
        return (bounds[:-1] + bounds[1:]) / 2

    @staticmethod
    def __boundsToCellLengths(bounds: np.ndarray) -> np.ndarray:
        """
        Convert cell bounds to cell lengths
        """
        return abs(bounds[:-1] - bounds[1:])

    @staticmethod
    def __boundsToDomainLength(bounds: np.ndarray):
        """
        Convert cell bounds to domain length
        """
        return abs(bounds[-1] - bounds[0])

    def __init__(self, grid: vtkRectilinearGrid):
        """
        The constructor for VTKGrid class

        Parameters:
            grid (vtk.vtkRectilinearGrid): the grid. Probably from VTKReader.getOutput()
        """
        self.vtk_grid = grid

        self.dims = list(self.vtk_grid.GetDimensions())

        # VTK dimensions are +1 from the number of cells
        self.dims = [d - 1 for d in self.dims]

    def getDimensions(self, d=None):
        """
        Number of cells in each direction

        Parameters:
            d (int): the direction

        Returns:
            int: The dimensions of the cell data, [Nx, Ny, Nz]. If d is provided, return Nd
        """
        if d is None:
            return self.dims
        else:
            return self.dims[d]

    # For getting coordinates
    def getXCoordinates(self):
        """
        Returns the x coordinates of cell centers
        """
        return VTKGrid.__boundsToCellCenter(
            vtk_to_numpy(self.vtk_grid.GetXCoordinates())
        )

    def getYCoordinates(self):
        """
        Returns the x coordinates of cell centers
        """
        return VTKGrid.__boundsToCellCenter(
            vtk_to_numpy(self.vtk_grid.GetYCoordinates())
        )

    def getZCoordinates(self):
        """
        Returns the x coordinates of cell centers
        """
        return VTKGrid.__boundsToCellCenter(
            vtk_to_numpy(self.vtk_grid.GetZCoordinates())
        )

    # For getting cell lengths
    def getDX(self):
        """
        Returns dx of cells
        """
        return VTKGrid.__boundsToCellLengths(
            vtk_to_numpy(self.vtk_grid.GetXCoordinates())
        )

    def getDY(self):
        """
        Returns dy of cells
        """
        return VTKGrid.__boundsToCellLengths(
            vtk_to_numpy(self.vtk_grid.GetYCoordinates())
        )

    def getDZ(self):
        """
        Returns dz of cells
        """
        return VTKGrid.__boundsToCellLengths(
            vtk_to_numpy(self.vtk_grid.GetZCoordinates())
        )

    # For getting domain lengths
    def getLX(self):
        """
        Returns length of the domain in X
        """
        return VTKGrid.__boundsToDomainLength(
            vtk_to_numpy(self.vtk_grid.GetXCoordinates())
        )

    def getLY(self):
        """
        Returns length of the domain in Y
        """
        return VTKGrid.__boundsToDomainLength(
            vtk_to_numpy(self.vtk_grid.GetYCoordinates())
        )

    def getLZ(self):
        """
        Returns length of the domain in Z
        """
        return VTKGrid.__boundsToDomainLength(
            vtk_to_numpy(self.vtk_grid.GetZCoordinates())
        )

    def getArray(self, name: str):
        """
        Get array data as a numpy array. If no such array exists, return None

        Parameters:
            name (str): the name of the data array

        Returns:
            A numpy array in order with number of dimensions depending on data type.
                Scalar: 3
                Vector: 4
                Tensor: 5
            The last three dimensions are [... i,j,k] corresponding to (x,y,z).
        """

        array = self.vtk_grid.GetCellData().GetArray(name)

        # check that the array exists
        if array is None:
            return None

        # convert to numpy
        comp = array.GetNumberOfComponents()
        if comp == 1:  # scalar
            return (
                vtk_to_numpy(array)
                .reshape(self.dims[2], self.dims[1], self.dims[0])
                .transpose(2, 1, 0)
            )
        elif comp == 3:  # vector
            return (
                vtk_to_numpy(array)
                .reshape(self.dims[2], self.dims[1], self.dims[0], 3)
                .transpose(3, 2, 1, 0)
            )
        elif comp == 9:  # tensor
            return (
                vtk_to_numpy(array)
                .reshape(self.dims[2], self.dims[1], self.dims[0], 3, 3)
                .transpose(3, 4, 2, 1, 0)
            )
        else:
            raise ValueError("Unsupported number of components in cell array:" + name)
