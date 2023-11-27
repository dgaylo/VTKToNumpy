import pytest
import os

from numpy import all
from vtk import vtkXMLRectilinearGridWriter
from conftest import addVectorData,addScalarData

from vtktonumpy import VTKReader

SCALAR_NAME="S"
VECTOR_NAME="V"

@pytest.fixture
def myVTKFile(myVtkRectGrid, myVectorData, myScalarData, tmp_path):
    # add scalar and vector data
    addVectorData(myVtkRectGrid,myVectorData,VECTOR_NAME)
    addScalarData(myVtkRectGrid,myScalarData,SCALAR_NAME)

    # figure out the file name
    target_output = os.path.join(tmp_path,'grid.vtr')

    writer = vtkXMLRectilinearGridWriter()
    writer.SetInputData(myVtkRectGrid)
    writer.SetFileName(target_output)
    writer.Update()

    return target_output

@pytest.fixture
def myVTKReader(myVTKFile) -> VTKReader:
    return VTKReader(myVTKFile)


def test_contains(myVTKReader):
    assert myVTKReader.contains(SCALAR_NAME)
    assert myVTKReader.contains(VECTOR_NAME)

# test initialization with array list
def test_Init2(myVTKFile):
    reader=VTKReader(myVTKFile,arrays=[SCALAR_NAME, VECTOR_NAME])
    assert reader.getArrayStatus() == [True, True]

# test getting array list
def test_getArrayList(myVTKReader):
    assert (
        myVTKReader.getArrayList() == [SCALAR_NAME, VECTOR_NAME] or
        myVTKReader.getArrayList() == [VECTOR_NAME, SCALAR_NAME] )

# test adding and removing arrays
def test_getArrayStatus(myVTKReader):
    assert myVTKReader.getArrayStatus() == [False, False]
    myVTKReader.addArray([SCALAR_NAME, VECTOR_NAME])
    assert myVTKReader.getArrayStatus() == [True, True]
    myVTKReader.removeArray([SCALAR_NAME, VECTOR_NAME])
    assert myVTKReader.getArrayStatus() == [False, False]

    # only turn on one
    myVTKReader.addArray([SCALAR_NAME])
    if myVTKReader.getArrayList() == [SCALAR_NAME, VECTOR_NAME]:
        assert myVTKReader.getArrayStatus() == [True, False]
    else:
        assert myVTKReader.getArrayStatus() == [False, True]

# test if reading grid returns correct values
def test_roundTrip(myVTKReader, myVectorData, myScalarData):
    myVTKReader.addArray([SCALAR_NAME, VECTOR_NAME])
    grid=myVTKReader.getGrid()

    # test scalar values
    s=grid.getArray(SCALAR_NAME)
    assert s.shape == myScalarData.shape
    assert all(s == myScalarData)

    # test vector values
    v=grid.getArray(VECTOR_NAME)
    assert v.shape == myVectorData.shape
    assert all(v == myVectorData)
