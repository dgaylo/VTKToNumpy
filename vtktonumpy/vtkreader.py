"""
    Contains VTKReader, a wrapper for VTK XLM Readers
"""

import os
import errno
from typing import Union
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
from vtkmodules.vtkIOXML import vtkXMLPRectilinearGridReader
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid

from .vtkgrid import VTKGrid


class VTKReader:
    """A wrapper for VTK XLM Readers"""

    @staticmethod
    def __getVtkReader(ext):
        if ext == "vtr":
            return vtkXMLRectilinearGridReader()
        if ext == "pvtr":
            return vtkXMLPRectilinearGridReader()

        raise SyntaxError("Unsupported File Type:" + ext)

    def __init__(self, filepath: os.PathLike, extension: str = "", arrays=None):
        """
        The constructor for VTKReader class

        Parameters:
            filepath (os.PathLike): the VTK file to read
            extension (str) : what type of file. Options are "vtr" or "pvtr"
            arrays : list of arrays to be read
        """

        # check that the file actually exists
        if not os.path.isfile(filepath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

        # figure out the extension
        if extension == "":
            extension = os.path.splitext(filepath)[1].replace(".", "")

        # Depending on extension, create the reader
        self.vtk_reader = VTKReader.__getVtkReader(extension.lower().strip())

        # Assign the file
        self.vtk_reader.SetFileName(filepath)
        self.vtk_reader.UpdateInformation()

        # Turn off all point arrays
        for i in range(self.vtk_reader.GetNumberOfPointArrays()):
            name = self.vtk_reader.GetPointArrayName(i)
            self.vtk_reader.SetPointArrayStatus(name, False)

        # Turn off all cell arrays, unless in arrays
        for i in range(self.vtk_reader.GetNumberOfCellArrays()):
            name = self.vtk_reader.GetCellArrayName(i)
            if arrays is not None:
                self.vtk_reader.SetCellArrayStatus(name, (name in arrays))
            else:
                self.vtk_reader.SetCellArrayStatus(name, False)

    def contains(self, array: str) -> bool:
        """
        Determine if the VTK data contains an array

        Parameters:
            array (str) : name of array

        Returns:
            bool: True if the array is in the VTK data.
        """
        for i in range(self.vtk_reader.GetNumberOfCellArrays()):
            if self.vtk_reader.GetCellArrayName(i) == array:
                return True

        return False

    def addArray(self, array: Union[str, list[str]]) -> None:
        """
        Add an array (or arrays) to those that will be read

        Parameters:
            array (str or list[str]) : name of array or arrays
        """
        if not isinstance(array, list):
            array = [array]

        for a in array:
            if self.contains(a):
                self.vtk_reader.SetCellArrayStatus(a, True)
            else:
                raise ValueError("Cell Array not present:" + a)

    def removeArray(self, array: Union[str, list[str]]) -> None:
        """
        Remove an array (or arrays) from those that will be read.
        If the array is not present in the VTK data, do nothing

        Parameters:
            array (str or list[str]) : name of array or arrays
        """
        if not isinstance(array, list):
            array = [array]

        for a in array:
            if self.contains(a):
                self.vtk_reader.SetCellArrayStatus(a, False)

    def getArrayList(self) -> list[str]:
        """
        Returns a list of arrays present in the VTK data

        Parameters:
            none

        Returns:
            out : a list of available cell arrays
        """
        out = []
        for i in range(self.vtk_reader.GetNumberOfCellArrays()):
            out.append(self.vtk_reader.GetCellArrayName(i))

        return out

    def getArrayStatus(self) -> list[bool]:
        """
        Returns read status of arrays present in the VTK data

        Parameters:
            none

        Returns:
            out : a boolean list of array status
        """
        out = []
        for i in range(self.vtk_reader.GetNumberOfCellArrays()):
            name = self.vtk_reader.GetCellArrayName(i)
            out.append(bool(self.vtk_reader.GetCellArrayStatus(name)))

        return out

    def getOutput(self) -> vtkRectilinearGrid:
        """
        A wrapper for VTK GetOutput() which includes Update()

        Returns:
            vtk.vtkRectilinearGrid: the reader's output
        """

        self.vtk_reader.Update()
        return self.vtk_reader.GetOutput()

    def getGrid(self) -> VTKGrid:
        """
        Returns the vtktonumpy vtkGrid generated by getOutput
        """
        return VTKGrid(self.getOutput())
