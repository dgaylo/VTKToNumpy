import pytest


from vtktonumpy.vtkgrid import VTKGrid
from numpy import all

@pytest.fixture
def myVTKGrid(myVtkRectGrid) -> VTKGrid:
    """
    Returns a VTKGrid based on the test vtkRectilinearGrid
    """
    return VTKGrid(myVtkRectGrid)

def test_getDimensions(myVTKGrid, myGrid):
    assert myVTKGrid.getDimensions(0) == myGrid.nx()
    assert myVTKGrid.getDimensions(1) == myGrid.ny()
    assert myVTKGrid.getDimensions(2) == myGrid.nz()
    assert myVTKGrid.getDimensions() == [myGrid.nx(),myGrid.ny(),myGrid.nz()]

# For getting coordinates
def test_getXCoordinates(myVTKGrid, myGrid):
    for i in range(myGrid.nx()):
        assert myVTKGrid.getXCoordinates()[i] == ( myGrid.xEdges[i+1] + myGrid.xEdges[i] ) / 2

def test_getYCoordinates(myVTKGrid, myGrid):
    for j in range(myGrid.ny()):
        assert myVTKGrid.getYCoordinates()[j] == ( myGrid.yEdges[j+1] + myGrid.yEdges[j] ) / 2

def test_getZCoordinates(myVTKGrid, myGrid):
    for k in range(myGrid.nz()):
        assert myVTKGrid.getZCoordinates()[k] == ( myGrid.zEdges[k+1] + myGrid.zEdges[k] ) / 2

# For getting cell lengths
def test_getDX(myVTKGrid, myGrid):
    for i in range(myGrid.nx()):
        assert myVTKGrid.getDX()[i] == myGrid.xEdges[i+1] - myGrid.xEdges[i]

def test_getDY(myVTKGrid, myGrid):
    for j in range(myGrid.ny()):
        assert myVTKGrid.getDY()[j] == myGrid.yEdges[j+1] - myGrid.yEdges[j]

def test_getDZ(myVTKGrid, myGrid):
    for k in range(myGrid.nz()):
        assert myVTKGrid.getDZ()[k] == myGrid.zEdges[k+1] - myGrid.zEdges[k]

# For getting domain lengths
def test_getLX(myVTKGrid, myGrid):
    assert myVTKGrid.getLX() == myGrid.xEdges[-1] - myGrid.xEdges[0]

def test_getLY(myVTKGrid, myGrid):
    assert myVTKGrid.getLY() ==  myGrid.yEdges[-1] - myGrid.yEdges[0]

def test_getLZ(myVTKGrid, myGrid):
    assert myVTKGrid.getLZ() == myGrid.zEdges[-1] - myGrid.zEdges[0]

# for testing domain extents
def test_getExtentsX(myVTKGrid, myGrid):
    assert myVTKGrid.getExtentsX() == (myGrid.xEdges[0], myGrid.xEdges[-1])

def test_getExtentsY(myVTKGrid, myGrid):
    assert myVTKGrid.getExtentsY() == (myGrid.yEdges[0], myGrid.yEdges[-1])

def test_getExtentsZ(myVTKGrid, myGrid):
    assert myVTKGrid.getExtentsZ() == (myGrid.zEdges[0], myGrid.zEdges[-1])

# test reading scalar data
def test_getArrayScalar(myVtkRectGrid_S,myScalarData):
    s=VTKGrid(myVtkRectGrid_S).getArray("S")
    assert s.shape == myScalarData.shape
    assert all(s == myScalarData)

# test reading vector data
def test_getArrayVector(myVtkRectGrid_V,myVectorData):
    v=VTKGrid(myVtkRectGrid_V).getArray("V")
    assert v.shape == myVectorData.shape
    assert all(v == myVectorData)
