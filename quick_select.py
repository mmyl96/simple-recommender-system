from random import *

class TopK:
    def __init__(self, array, k):
        self.flag = 1
        self.ori_array = array
        self.array = abs(array)
        self.k = k
        self.left = 0
        self.right = len(array)-1
        self.index_array = [i for i in range(len(array))]

    # Swap the pointers
    # Includes index
    def swap(self, i, j):
        self.array[i], self.array[j] = self.array[j], self.array[i]
        self.index_array[i], self.index_array[j] = self.index_array[j], self.index_array[i]
        self.ori_array[i], self.ori_array[j] = self.ori_array[j], self.ori_array[i]

    # Same as partition in quick sort
    # But in here, larger numbers are on the head
    def partition(self, pivotIndex):
        pivotValue = self.array[pivotIndex]
        self.swap(pivotIndex, self.right)
        tempIndex = self.left
        for i in range(self.left, self.right):
            if self.array[i] > pivotValue:
                self.swap(tempIndex, i)
                tempIndex += 1
            if self.array[i] == pivotValue:
                if self.flag == 1:
                    self.swap(tempIndex, i)
                    tempIndex += 1
                    self.flag = 0
                else:
                    self.flag = 1
        self.swap(self.right, tempIndex)
        return tempIndex

    # Return K largest results and their indices
    # instead of Kth largest result
    # A bit difference with quick select
    def answer(self):
        if self.k > len(self.array):
            return "There is no enough elements!", None
        if self.left == self.right:
            return self.ori_array[:self.k], self.index_array[:self.k]
        pivotIndex = randint(self.left, self.right)
        pivotIndex = self.partition(pivotIndex)
        if self.k-1 == pivotIndex:
            return self.ori_array[:self.k], self.index_array[:self.k]
        elif self.k-1 < pivotIndex:
            self.right = pivotIndex-1
        else:
            self.left = pivotIndex+1
        return self.answer()
