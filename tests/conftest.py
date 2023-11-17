from dataclasses import dataclass
from math import sin
import pytest



from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
from vtkmodules.util.numpy_support import numpy_to_vtk

from numpy import random, ndarray

def addScalarData(rgrid: vtkRectilinearGrid, data: ndarray, name: str) -> vtkRectilinearGrid:
    s=data.shape
    array=numpy_to_vtk(data.transpose(2, 1, 0).reshape(s[0]*s[1]*s[2]),deep=True)
    array.SetName(name)
    return rgrid.GetCellData().AddArray(array)

def addVectorData(rgrid: vtkRectilinearGrid, data: ndarray, name: str) -> vtkRectilinearGrid:
    s=data.shape
    array=numpy_to_vtk(data.transpose(3, 2, 1, 0).reshape(s[1]*s[2]*s[3],s[0]),deep=True)
    array.SetName(name)
    return rgrid.GetCellData().AddArray(array)

def addTensorData(rgrid: vtkRectilinearGrid, data: ndarray, name: str) -> vtkRectilinearGrid:
    s=data.shape
    array=numpy_to_vtk(data.transpose(4, 3, 2, 0, 1).reshape(s[2]*s[3]*s[4],s[0]*s[1]),deep=True)
    array.SetName(name)
    return rgrid.GetCellData().AddArray(array)

@dataclass
class GridClass:
    """
    Class for storing truth about a grid
    """
    xEdges: list
    yEdges: list
    zEdges: list

    def nx(self):
        return len(self.xEdges)-1

    def ny(self):
        return len(self.yEdges)-1

    def nz(self):
        return len(self.zEdges)-1

@pytest.fixture(scope="session")
def myGrid() -> GridClass:
    # Based on: https://examples.vtk.org/site/Python/RectilinearGrid/RGrid/
    return GridClass(
        xEdges = [
            -1.22396,
            -1.17188,
            -1.11979,
            -1.06771,
            -1.01562,
            -0.963542,
            -0.911458,
            -0.859375,
            -0.807292,
            -0.755208,
            -0.703125,
            -0.651042,
            -0.598958,
            -0.546875,
            -0.494792,
            -0.442708,
            -0.390625,
            -0.338542,
            -0.286458,
            -0.234375,
            -0.182292,
            -0.130209,
            -0.078125,
            -0.026042,
            0.0260415,
            0.078125,
            0.130208,
            0.182291,
            0.234375,
            0.286458,
            0.338542,
            0.390625,
            0.442708,
            0.494792,
            0.546875,
            0.598958,
            0.651042,
            0.703125,
            0.755208,
            0.807292,
            0.859375,
            0.911458,
            0.963542,
            1.01562,
            1.06771,
            1.11979,
            1.17188,
        ],
        yEdges = [
            -1.25,
            -1.17188,
            -1.09375,
            -1.01562,
            -0.9375,
            -0.859375,
            -0.78125,
            -0.703125,
            -0.625,
            -0.546875,
            -0.46875,
            -0.390625,
            -0.3125,
            -0.234375,
            -0.15625,
            -0.078125,
            0,
            0.078125,
            0.15625,
            0.234375,
            0.3125,
            0.390625,
            0.46875,
            0.546875,
            0.625,
            0.703125,
            0.78125,
            0.859375,
            0.9375,
            1.01562,
            1.09375,
            1.17188,
            1.25,
        ],
        zEdges = [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            1,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.75,
            1.8,
            1.9,
            2,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.75,
            2.8,
            2.9,
            3,
            3.1,
            3.2,
            3.3,
            3.4,
            3.5,
            3.6,
            3.7,
            3.75,
            3.8,
            3.9,
        ]
    )

@pytest.fixture
def myVtkRectGrid(myGrid) -> vtkRectilinearGrid:
    """
    Returns a test vtkRectilinearGrid
    Based on: https://examples.vtk.org/site/Python/RectilinearGrid/RGrid/
    """
    # Create a rectilinear grid by defining three arrays specifying the
    # coordinates in the x-y-z directions.
    xCoords = vtkDoubleArray()
    for x in myGrid.xEdges:
        xCoords.InsertNextValue(x)

    yCoords = vtkDoubleArray()
    for y in myGrid.yEdges:
        yCoords.InsertNextValue(y)

    zCoords = vtkDoubleArray()
    for z in myGrid.zEdges:
        zCoords.InsertNextValue(z)

    rgrid = vtkRectilinearGrid()
    rgrid.SetDimensions(myGrid.nx()+1, myGrid.ny()+1, myGrid.nz()+1)
    rgrid.SetXCoordinates(xCoords)
    rgrid.SetYCoordinates(yCoords)
    rgrid.SetZCoordinates(zCoords)

    return rgrid

@pytest.fixture(scope="session")
def myScalarData(myGrid) -> ndarray:
    return (random.rand(myGrid.nx(),myGrid.ny(),myGrid.nz()) - 0.5)*10

@pytest.fixture(scope="session")
def myVectorData(myGrid) -> ndarray:
    return (random.rand(3,myGrid.nx(),myGrid.ny(),myGrid.nz()) - 0.5)*10

@pytest.fixture(scope="session")
def myTensorData(myGrid) -> ndarray:
    return (random.rand(3,3,myGrid.nx(),myGrid.ny(),myGrid.nz()) - 0.5)*10

@pytest.fixture
def myVtkRectGrid_S(myVtkRectGrid, myScalarData) -> vtkRectilinearGrid:
    addScalarData(myVtkRectGrid,myScalarData,"S")
    return myVtkRectGrid

@pytest.fixture
def myVtkRectGrid_V(myVtkRectGrid, myVectorData) -> vtkRectilinearGrid:
    addVectorData(myVtkRectGrid,myVectorData,"V")
    return myVtkRectGrid

@pytest.fixture
def myVtkRectGrid_T(myVtkRectGrid, myTensorData) -> vtkRectilinearGrid:
    addTensorData(myVtkRectGrid,myTensorData,"T")
    return myVtkRectGrid
