import pytest

from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid

from vtktonumpy.vtkgrid import VTKGrid


# Based on: https://examples.vtk.org/site/Python/RectilinearGrid/RGrid/
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
]
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
]
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

nx = len(xEdges) - 1
ny = len(yEdges) - 1
nz = len(zEdges) - 1

@pytest.fixture(scope="session")
def myVtkRectGrid() -> vtkRectilinearGrid:
    """
    Returns a test vtkRectilinearGrid
    Based on: https://examples.vtk.org/site/Python/RectilinearGrid/RGrid/
    """
    x = xEdges
    y = yEdges
    z = zEdges

    # Create a rectilinear grid by defining three arrays specifying the
    # coordinates in the x-y-z directions.
    xCoords = vtkDoubleArray()
    for i in range(0, len(x)):
        xCoords.InsertNextValue(x[i])
    yCoords = vtkDoubleArray()
    for i in range(0, len(y)):
        yCoords.InsertNextValue(y[i])
    zCoords = vtkDoubleArray()
    for i in range(0, len(z)):
        zCoords.InsertNextValue(z[i])

    rgrid = vtkRectilinearGrid()
    rgrid.SetDimensions(len(x), len(y), len(z))
    rgrid.SetXCoordinates(xCoords)
    rgrid.SetYCoordinates(yCoords)
    rgrid.SetZCoordinates(zCoords)

    return rgrid

@pytest.fixture(scope="session")
def myVTKGrid(myVtkRectGrid) -> VTKGrid:
    """
    Returns a VTKGrid based on the test vtkRectilinearGrid
    """
    return VTKGrid(myVtkRectGrid)

def test_getDimensions(myVTKGrid):
    assert myVTKGrid.getDimensions(0) == nx
    assert myVTKGrid.getDimensions(1) == ny
    assert myVTKGrid.getDimensions(2) == nz
    assert myVTKGrid.getDimensions() == [nx,ny,nz]

# For getting coordinates
@pytest.mark.parametrize("i", range(nx))
def test_getXCoordinates(myVTKGrid,i):
    assert myVTKGrid.getXCoordinates()[i] == ( xEdges[i+1] + xEdges[i] ) / 2

@pytest.mark.parametrize("j", range(ny))
def test_getYCoordinates(myVTKGrid, j):
    assert myVTKGrid.getYCoordinates()[j] == ( yEdges[j+1] + yEdges[j] ) / 2

@pytest.mark.parametrize("k", range(nz))
def test_getZCoordinates(myVTKGrid, k):
    assert myVTKGrid.getZCoordinates()[k] == ( zEdges[k+1] + zEdges[k] ) / 2

# For getting cell lengths
@pytest.mark.parametrize("i", range(nx))
def test_getDX(myVTKGrid,i):
    assert myVTKGrid.getDX()[i] == xEdges[i+1] - xEdges[i]

@pytest.mark.parametrize("j", range(ny))
def test_getDY(myVTKGrid, j):
    assert myVTKGrid.getDY()[j] == yEdges[j+1] - yEdges[j]

@pytest.mark.parametrize("k", range(nz))
def test_getDZ(myVTKGrid, k):
    assert myVTKGrid.getDZ()[k] == zEdges[k+1] - zEdges[k]

    diff = zEdges[k+1] - zEdges[k]
    assert myVTKGrid.getDZ()[k] == diff

# For getting domain lengths
def test_getLX(myVTKGrid):
    assert myVTKGrid.getLX() == xEdges[-1] - xEdges[0]

def test_getLY(myVTKGrid):
    assert myVTKGrid.getLY() ==  yEdges[-1] - yEdges[0]

def test_getLZ(myVTKGrid):
    assert myVTKGrid.getLZ() == zEdges[-1] - zEdges[0]

# for testing domain extents
def test_getExtentsX(myVTKGrid):
    assert myVTKGrid.getExtentsX() == (xEdges[0], xEdges[-1])
