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
    def __boundsToDomainLength(bounds: np.ndarray) -> float:
        """
        Convert cell bounds to domain length
        """
        return abs(bounds[-1] - bounds[0])

    @staticmethod
    def __boundsToDomainExtents(bounds: np.ndarray) -> tuple:
        """
        Convert cell bounds to domain length
        """
        return (bounds[0], bounds[-1])

    def __init__(self, grid: vtkRectilinearGrid):
        """
        The constructor for VTKGrid class

        Args:
            grid (vtk.vtkRectilinearGrid): The underlying VTK rectilinear grid,
                such as from [VTKReader.getOutput()](vtkreader.html#VTKReader.getOutput).
        """
        self.vtk_grid = grid
        """Underlying VTK rectilinear grid"""

        self.dims = list(self.vtk_grid.GetDimensions())

        # VTK dimensions are +1 from the number of cells
        self.dims = [d - 1 for d in self.dims]
        """$[N_x,N_y,N_z]$"""

    def getDimensions(self, d=None):
        """
        Get number of cells in each direction.

        Parameters:
            d (:obj:`int`, optional): the direction

        Returns:
            int[3]: The dimensions of the cell data, $[N_x, N_y, N_z]$.

            int: If :obj:`d` is set, returns $[N_d]$.
        """
        if d is None:
            return self.dims
        else:
            return self.dims[d]

    def getNX(self) -> int:
        """
        $N_x$, the number of cells in $x$ direction
        """
        return self.dims[0]

    def getNY(self) -> int:
        """
        $N_y$, the number of cells in $y$ direction
        """
        return self.dims[1]

    def getNZ(self) -> int:
        """
        $N_z$, the number of cells in $z$ direction
        """
        return self.dims[2]

    # For getting coordinates
    def getXCoordinates(self) -> np.ndarray:
        """
        The $x$-coordinates of cell centers
        """
        return VTKGrid.__boundsToCellCenter(
            vtk_to_numpy(self.vtk_grid.GetXCoordinates())
        )

    def getYCoordinates(self) -> np.ndarray:
        """
        The $y$-coordinates of cell centers
        """
        return VTKGrid.__boundsToCellCenter(
            vtk_to_numpy(self.vtk_grid.GetYCoordinates())
        )

    def getZCoordinates(self) -> np.ndarray:
        """
        The $z$-coordinates of cell centers
        """
        return VTKGrid.__boundsToCellCenter(
            vtk_to_numpy(self.vtk_grid.GetZCoordinates())
        )

    # For getting cell lengths
    def getDX(self) -> np.ndarray:
        r"""
        The $\Delta x$ of cells
        """
        return VTKGrid.__boundsToCellLengths(
            vtk_to_numpy(self.vtk_grid.GetXCoordinates())
        )

    def getDY(self) -> np.ndarray:
        r"""
        The $\Delta y$ of cells
        """
        return VTKGrid.__boundsToCellLengths(
            vtk_to_numpy(self.vtk_grid.GetYCoordinates())
        )

    def getDZ(self) -> np.ndarray:
        r"""
        The $\Delta z$ of cells
        """
        return VTKGrid.__boundsToCellLengths(
            vtk_to_numpy(self.vtk_grid.GetZCoordinates())
        )

    # For getting domain lengths
    def getLX(self) -> float:
        """
        $L_x$, the length of the domain in $x$
        """
        return VTKGrid.__boundsToDomainLength(
            vtk_to_numpy(self.vtk_grid.GetXCoordinates())
        )

    def getLY(self) -> float:
        """
        $L_y$, the length of the domain in $y$
        """
        return VTKGrid.__boundsToDomainLength(
            vtk_to_numpy(self.vtk_grid.GetYCoordinates())
        )

    def getLZ(self) -> float:
        """
        $L_z$, the length of the domain in $z$
        """
        return VTKGrid.__boundsToDomainLength(
            vtk_to_numpy(self.vtk_grid.GetZCoordinates())
        )

    # For getting domain bounds
    def getExtentsX(self) -> tuple:
        r"""
        $(x_{\mathrm{start}},x_{\mathrm{end}})$, the extents in $x$.
        """
        return VTKGrid.__boundsToDomainExtents(
            vtk_to_numpy(self.vtk_grid.GetXCoordinates())
        )

    def getExtentsY(self) -> tuple:
        r"""
        $(y_{\mathrm{start}},y_{\mathrm{end}})$, the extents in $y$.
        """
        return VTKGrid.__boundsToDomainExtents(
            vtk_to_numpy(self.vtk_grid.GetYCoordinates())
        )

    def getExtentsZ(self) -> tuple:
        r"""
        $(z_{\mathrm{start}},z_{\mathrm{end}})$, the extents in $z$.
        """
        return VTKGrid.__boundsToDomainExtents(
            vtk_to_numpy(self.vtk_grid.GetZCoordinates())
        )

    # For data arrays
    def getArrayList(self):
        """
        list[str]: A list of cell arrays present in the VTK data
        """
        out = []
        cell_data=self.vtk_grid.GetCellData()
        for i in range(cell_data.GetNumberOfArrays()):
            out.append(cell_data.GetArrayName(i))

        return out

    def getArray(self, name: str) -> np.ndarray:
        r"""
        Get VTK array as a numpy array. If no such array exists, return :obj:`None`

        Parameters:
            name (str): the name of the array

        Returns:
            A numpy array, with the number of array dimensions depending on data type:

                Scalar: ndim=3

                Vector: ndim=4

                Tensor: ndim=5

            The last three dimensions are $[\dots,i,j,k]$ corresponding to $(x,y,z)$.
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
        if comp == 3:  # vector
            return (
                vtk_to_numpy(array)
                .reshape(self.dims[2], self.dims[1], self.dims[0], 3)
                .transpose(3, 2, 1, 0)
            )
        if comp == 9:  # tensor
            return (
                vtk_to_numpy(array)
                .reshape(self.dims[2], self.dims[1], self.dims[0], 3, 3)
                .transpose(3, 4, 2, 1, 0)
            )

        raise ValueError("Unsupported number of components in cell array:" + name)
